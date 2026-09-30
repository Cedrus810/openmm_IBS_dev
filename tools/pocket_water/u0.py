"""Cheap pairwise water energy U0 for the relocation sampler.

U0 is the direct-space part of the production nonbonded energy between
different molecules: erfc-damped Coulomb (same Ewald alpha and cutoff as the
production PME) + Lennard-Jones with the same cutoff (Lorentz-Berthelot, no
switch).  It is a well-defined pairwise-additive function of the full
configuration, so moving one water changes U0 by exactly u0(new) - u0(old),
where u0 is that water's interaction with everything else.  The reciprocal
part and the dispersion correction are left out on purpose; delayed
acceptance with the exact OpenMM energy puts them back (relocation.py).

Orthorhombic boxes only.  All lengths nm, energies kJ/mol.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc
from scipy.spatial import cKDTree

ONE_4PI_EPS0 = 138.935456  # kJ mol^-1 nm e^-2 (OpenMM's value)


class U0Engine:
    def __init__(self, charges, sigmas, epsilons, cutoff, alpha, water_atoms):
        """water_atoms: (n_water, 3) atom indices, oxygen first."""
        self.q = np.asarray(charges, dtype=np.float64)
        self.sig = np.asarray(sigmas, dtype=np.float64)
        self.eps = np.asarray(epsilons, dtype=np.float64)
        self.rc = float(cutoff)
        self.alpha = float(alpha)
        self.water_atoms = np.asarray(water_atoms, dtype=np.int64)
        n = len(self.q)
        self.mol_of = np.full(n, -1, dtype=np.int64)  # water index of each atom, -1 if not water
        for w, idx in enumerate(self.water_atoms):
            self.mol_of[idx] = w
        self.box = None
        self.pos = None
        self.tree = None

    @classmethod
    def from_openmm(cls, nonbonded_force, context, water_atoms):
        nb = nonbonded_force
        from openmm import unit
        q, s, e = [], [], []
        for i in range(nb.getNumParticles()):
            qi, si, ei = nb.getParticleParameters(i)
            q.append(qi.value_in_unit(unit.elementary_charge))
            s.append(si.value_in_unit(unit.nanometer))
            e.append(ei.value_in_unit(unit.kilojoule_per_mole))
        alpha, _, _, _ = nb.getPMEParametersInContext(context)
        if hasattr(alpha, "value_in_unit"):
            alpha = alpha.value_in_unit(unit.nanometer ** -1)
        return cls(q, s, e, nb.getCutoffDistance().value_in_unit(unit.nanometer), alpha, water_atoms)

    # ------------------------------------------------------------ state
    def set_state(self, positions, box_lengths):
        self.box = np.asarray(box_lengths, dtype=np.float64)
        self.pos = np.asarray(positions, dtype=np.float64)
        self.tree = cKDTree(self._wrap(self.pos), boxsize=self.box)

    def _wrap(self, x):
        w = np.mod(x, self.box)
        w[w >= self.box] = 0.0
        return w

    # ------------------------------------------------------------ pairs
    def _pair_energy(self, qa, sa, ea, j, r):
        qb, sb, eb = self.q[j], self.sig[j], self.eps[j]
        elec = ONE_4PI_EPS0 * qa * qb * erfc(self.alpha * r) / r
        sg = 0.5 * (sa + sb)
        ep = np.sqrt(ea * eb)
        sr6 = (sg / r) ** 6
        return elec + 4.0 * ep * (sr6 * sr6 - sr6)

    def probe_pairs(self, sites, site_params, exclude_mol=None):
        """Pairs between probe sites and the current configuration.

        sites: (m, 3) positions; site_params: (q, sigma, eps) arrays of length m.
        Returns (probe_row, atom_j, energy) for every pair within the cutoff,
        excluding atoms of water `exclude_mol` (array per row or scalar)."""
        pt = cKDTree(self._wrap(np.asarray(sites, dtype=np.float64)), boxsize=self.box)
        sp = pt.sparse_distance_matrix(self.tree, self.rc, output_type="ndarray")
        i, j, r = sp["i"].astype(np.int64), sp["j"].astype(np.int64), sp["v"]
        keep = r > 0.0
        if exclude_mol is not None:
            ex = np.broadcast_to(np.asarray(exclude_mol), (len(site_params[0]),))
            keep &= self.mol_of[j] != ex[i]
        i, j, r = i[keep], j[keep], r[keep]
        qa, sa, ea = (np.asarray(p)[i] for p in site_params)
        return i, j, self._pair_energy(qa, sa, ea, j, r)

    def water_energies(self, waters):
        """u0 of each listed water (interaction with every atom not in it)."""
        waters = np.asarray(waters, dtype=np.int64)
        atoms = self.water_atoms[waters].reshape(-1)
        rows_mol = np.repeat(waters, 3)
        i, j, e = self.probe_pairs(self.pos[atoms], (self.q[atoms], self.sig[atoms], self.eps[atoms]),
                                   exclude_mol=rows_mol)
        return np.bincount(i // 3, weights=e, minlength=len(waters))

    def water_pair_matrix(self, waters_a):
        """phi(a, w) for every water a in waters_a and every other water w within the
        cutoff.  Returns dict a_index -> (w_array, phi_array)."""
        waters_a = np.asarray(waters_a, dtype=np.int64)
        atoms = self.water_atoms[waters_a].reshape(-1)
        rows_mol = np.repeat(waters_a, 3)
        i, j, e = self.probe_pairs(self.pos[atoms], (self.q[atoms], self.sig[atoms], self.eps[atoms]),
                                   exclude_mol=rows_mol)
        wj = self.mol_of[j]
        m = wj >= 0
        a = i[m] // 3
        key = a * len(self.water_atoms) + wj[m]
        uk, inv = np.unique(key, return_inverse=True)
        phi = np.bincount(inv, weights=e[m])
        out = {}
        aa = uk // len(self.water_atoms)
        ww = uk % len(self.water_atoms)
        for k in range(len(waters_a)):
            sel = aa == k
            out[int(waters_a[k])] = (ww[sel], phi[sel])
        return out
