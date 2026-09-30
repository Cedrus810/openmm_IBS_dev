"""Fixed-total-water relocation MC between a pocket region R and a shell S.

Move (JCTC 2026 SwapMC, eq 1 selection; our acceptance):
  * direction in (S -> R) or out (R -> S), probability 1/2 each
  * source water chosen with p_i = exp(beta u_i) / Z_src, u_i = its U0 interaction
  * new oxygen position uniform in the destination region, uniform random
    orientation (rigid geometry of the source water, rotated about O)

Stage 1 (Metropolis on pi0 ~ exp(-beta U0 + g(N))):
  in : A1 = (V_R / V_S) * Z_S(x) / Z_R(y) * exp(g(N+1) - g(N))
  out: A1 = (V_S / V_R) * Z_R(x) / Z_S(y) * exp(g(N-1) - g(N))
  Z_X(state) = sum over waters whose O is in X of exp(beta u), evaluated in
  that state (so Z_R(y) contains the moved water and the neighbours'
  interaction changes).  The moved water's own energy cancels.
Stage 2 (delayed acceptance, exact Hamiltonian U):
  A2 = min(1, exp(-beta [(U(y) - U(x)) - (U0(y) - U0(x))]))
Together the move is reversible with respect to exp(-beta U + g(N)).

Every MC phase performs a fixed number of attempts.  Attempts are evaluated
in batches from the current state and scanned in order; after an accepted
move the rest of the batch is discarded and regenerated from the new state,
so the phase is exactly a fixed-length sequence of Metropolis steps.

R = union of spheres (radius r_in) around reference centres, S = union of
spheres (radius r_out) minus R.  Both follow the protein rigidly through
`centers_fn(positions)`, so V_R and V_S are constants.
"""

from __future__ import annotations

import numpy as np


def logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = a.max()
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.exp(a - m).sum()))


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def min_image(d, box):
    return d - box * np.round(d / box)


def min_dist(points, centers, box):
    d = min_image(points[:, None, :] - centers[None, :, :], box)
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d)).min(1)


def union_volume(centers, r, rng, n=2_000_000, exclude=None):
    """Volume of union(centers, r) minus union(centers, exclude) (same centres)."""
    lo, hi = centers.min(0) - r, centers.max(0) + r
    pts = lo + (hi - lo) * rng.random((n, 3))
    d = np.sqrt(((pts[:, None, :] - centers[None]) ** 2).sum(-1)).min(1)
    inside = d < r
    if exclude is not None:
        inside &= d >= exclude
    return float(np.prod(hi - lo) * inside.mean())


def uniform_in_shell(centers, r_lo, r_hi, rng, n):
    """Uniform points with r_lo <= dist-to-nearest-centre < r_hi (r_lo = 0: plain union)."""
    lo, hi = centers.min(0) - r_hi, centers.max(0) + r_hi
    out, got = [], 0
    while got < n:
        pts = lo + (hi - lo) * rng.random((max(4 * n, 1000), 3))
        d = np.sqrt(((pts[:, None, :] - centers[None]) ** 2).sum(-1)).min(1)
        sel = pts[(d < r_hi) & (d >= r_lo)]
        out.append(sel)
        got += len(sel)
    return np.concatenate(out)[:n]


class RelocationMC:
    def __init__(self, engine, beta, r_in, r_out, centers_ref, centers_fn, exact_energy_fn,
                 rng, g=None, volume_samples=2_000_000):
        """engine: U0Engine.  centers_fn(positions) -> current sphere centres.
        exact_energy_fn(positions) -> exact potential energy (kJ/mol).
        g: None or callable/array giving the N bias (in kT units, added to ln pi)."""
        self.eng = engine
        self.beta = float(beta)
        self.r_in, self.r_out = float(r_in), float(r_out)
        self.centers_fn = centers_fn
        self.exact = exact_energy_fn
        self.rng = rng
        self.g = g
        self.V_R = union_volume(centers_ref, r_in, rng, volume_samples)
        self.V_S = union_volume(centers_ref, r_out, rng, volume_samples, exclude=r_in)
        self.stats = {"attempts": 0, "s1_in": 0, "s1_out": 0, "acc_in": 0, "acc_out": 0,
                      "s2_log_alpha": []}

    def _g(self, n):
        if self.g is None:
            return 0.0
        if callable(self.g):
            return float(self.g(n))
        return float(self.g[n]) if 0 <= n < len(self.g) else -np.inf

    # ------------------------------------------------------------ state
    def setup(self, positions, box):
        eng = self.eng
        eng.set_state(positions, box)
        self.pos = np.asarray(positions, dtype=np.float64).copy()
        self.box = np.asarray(box, dtype=np.float64)
        self.centers = self.centers_fn(self.pos)
        O = eng.water_atoms[:, 0]
        d = min_dist(self.pos[O], self.centers, self.box)
        self.R = np.where(d < self.r_in)[0]
        self.S = np.where((d >= self.r_in) & (d < self.r_out))[0]
        u = eng.water_energies(np.concatenate([self.R, self.S]))
        self.u_R, self.u_S = u[:len(self.R)], u[len(self.R):]
        self.logZ_R = logsumexp(self.beta * self.u_R)
        self.logZ_S = logsumexp(self.beta * self.u_S)
        # phi between each R water and every water within the cutoff
        self.phiR = eng.water_pair_matrix(self.R) if len(self.R) else {}
        self.S_index = {int(w): k for k, w in enumerate(self.S)}
        self.R_index = {int(w): k for k, w in enumerate(self.R)}
        # dense phi(R_k, S_w)
        self.phi_RS = np.zeros((len(self.R), len(self.S)))
        for k, w in enumerate(self.R):
            ws, ph = self.phiR[int(w)]
            for ww, p in zip(ws, ph):
                j = self.S_index.get(int(ww))
                if j is not None:
                    self.phi_RS[k, j] = p
        self.E_exact = self.exact(self.pos)

    @property
    def N(self):
        return len(self.R)

    # ------------------------------------------------------------ trials
    def propose(self, n):
        """Random proposals from the current state: (is_in, source_water, sites (n,3,3))."""
        rng = self.rng
        is_in = rng.random(n) < 0.5
        src = np.full(n, -1, dtype=np.int64)
        n_in, n_out = int(is_in.sum()), int((~is_in).sum())
        if n_in and len(self.S):
            p = np.exp(self.beta * self.u_S - self.logZ_S)
            src[is_in] = self.S[rng.choice(len(self.S), size=n_in, p=p / p.sum())]
        if n_out and len(self.R):
            p = np.exp(self.beta * self.u_R - self.logZ_R)
            src[~is_in] = self.R[rng.choice(len(self.R), size=n_out, p=p / p.sum())]
        dest = np.empty((n, 3))
        if n_in:
            dest[is_in] = uniform_in_shell(self.centers, 0.0, self.r_in, rng, n_in)
        if n_out:
            dest[~is_in] = uniform_in_shell(self.centers, self.r_in, self.r_out, rng, n_out)
        rots = random_rotations(rng, n)
        sites = np.empty((n, 3, 3))
        valid = src >= 0
        atoms = self.eng.water_atoms[np.where(valid, src, 0)]
        geom = min_image(self.pos[atoms] - self.pos[atoms[:, :1]], self.box)
        sites[:] = dest[:, None, :] + np.einsum("nij,nkj->nki", rots, geom)
        return is_in, src, sites, valid

    def evaluate(self, is_in, src, sites, valid=None, clip=True):
        """Stage-1 log acceptance and Delta U0 for explicit trials (clip=False
        returns the log Metropolis ratio before min(0, .))."""
        eng = self.eng
        n = len(src)
        if valid is None:
            valid = src >= 0
        atoms = eng.water_atoms[np.where(valid, src, 0)]
        params = (eng.q[atoms].reshape(-1), eng.sig[atoms].reshape(-1), eng.eps[atoms].reshape(-1))
        rows_mol = np.repeat(np.where(valid, src, -2), 3)
        i, j, e = eng.probe_pairs(sites.reshape(-1, 3), params, exclude_mol=rows_mol)
        t = i // 3
        u_y = np.bincount(t, weights=e, minlength=n)
        wj = eng.mol_of[j]
        # per (trial, water) interaction of the moved water at its new place
        m = wj >= 0
        nw = len(eng.water_atoms)
        key = t[m] * nw + wj[m]
        uk, inv = np.unique(key, return_inverse=True)
        phi_new = np.bincount(inv, weights=e[m])
        tt, ww = uk // nw, uk % nw
        order = np.argsort(tt, kind="stable")
        tt, ww, phi_new = tt[order], ww[order], phi_new[order]
        starts = np.searchsorted(tt, np.arange(n + 1))

        # source's own energy in the current state
        u_src = np.zeros(n)
        for k in range(n):
            if not valid[k]:
                continue
            s = int(src[k])
            u_src[k] = self.u_S[self.S_index[s]] if is_in[k] else self.u_R[self.R_index[s]]

        log_a = np.full(n, -np.inf)
        dU0 = u_y - u_src
        b = self.beta
        lnV = np.log(self.V_R / self.V_S)
        N = self.N
        for k in range(n):
            if not valid[k]:
                continue
            s = int(src[k])
            w_k = ww[starts[k]:starts[k + 1]]
            p_k = phi_new[starts[k]:starts[k + 1]]
            if is_in[k]:
                # pocket waters: + phi(new) - phi(source at old place)
                uR = self.u_R.copy()
                if len(self.R):
                    uR -= self.phi_RS[:, self.S_index[s]]
                    for w, p in zip(w_k, p_k):
                        r = self.R_index.get(int(w))
                        if r is not None:
                            uR[r] += p
                logZ_R_y = logsumexp(np.concatenate([b * uR, [b * u_y[k]]]))
                log_a[k] = lnV + self.logZ_S - logZ_R_y + self._g(N + 1) - self._g(N)
            else:
                # shell waters touched by the source leaving or by the new position
                ws_old, ph_old = self.phiR[s]
                delta = {}
                for w, p in zip(ws_old, ph_old):
                    if int(w) in self.S_index:
                        delta[int(w)] = delta.get(int(w), 0.0) - p
                for w, p in zip(w_k, p_k):
                    if int(w) in self.S_index:
                        delta[int(w)] = delta.get(int(w), 0.0) + p
                if delta:
                    idx = np.array([self.S_index[w] for w in delta])
                    dv = np.array(list(delta.values()))
                    new_terms = b * (self.u_S[idx] + dv)
                    rest_mask = np.ones(len(self.S), bool)
                    rest_mask[idx] = False
                    terms = np.concatenate([b * self.u_S[rest_mask], new_terms, [b * u_y[k]]])
                else:
                    terms = np.concatenate([b * self.u_S, [b * u_y[k]]])
                logZ_S_y = logsumexp(terms)
                log_a[k] = -lnV + self.logZ_R - logZ_S_y + self._g(N - 1) - self._g(N)
        return (np.minimum(log_a, 0.0) if clip else log_a), dU0, u_y

    # ------------------------------------------------------------ phase
    def run_phase(self, positions, box, n_attempts, batch=2000):
        """Fixed number of attempts; returns new positions and moved waters."""
        self.setup(positions, box)
        moved = []
        done = 0
        rng = self.rng
        while done < n_attempts:
            n = min(batch, n_attempts - done)
            is_in, src, sites, valid = self.propose(n)
            log_a, dU0, _ = self.evaluate(is_in, src, sites, valid)
            logu = np.log(rng.random(n))
            used = n
            for k in np.where(logu < log_a)[0]:
                if is_in[k]:
                    self.stats["s1_in"] += 1
                else:
                    self.stats["s1_out"] += 1
                trial = self.pos.copy()
                trial[self.eng.water_atoms[src[k]]] = sites[k]
                E_y = self.exact(trial)
                la2 = min(0.0, -self.beta * ((E_y - self.E_exact) - dU0[k]))
                self.stats["s2_log_alpha"].append(la2)
                if np.log(rng.random()) < la2:
                    self.stats["acc_in" if is_in[k] else "acc_out"] += 1
                    moved.append(int(src[k]))
                    used = int(k) + 1
                    self.setup(trial, self.box)
                    break
            self.stats["attempts"] += used
            done += used
        return self.pos, moved
