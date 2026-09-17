"""静态遍历 Stage-2 控制器：死动作 / 死出口 / 死代码 / 不可达守卫。"""
import ast, sys, collections
SRC="/home/ruigengji/ABFE_IBS/ABFE_IBS/abfe_preoptimizer.py"
PIPE="/home/ruigengji/ABFE_IBS/ABFE_IBS/abfe_pipeline.py"
tree=ast.parse(open(SRC).read()); ptree=ast.parse(open(PIPE).read())

cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)
         and n.name=="Stage2RepairController")
def const_tuple(name):
    for n in cls.body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","")==name:
            return [e.value for e in n.value.elts if isinstance(e,ast.Constant)]
    return []
ACTIONS=const_tuple("ACTIONS"); EXITS=const_tuple("EXITS"); TERM=const_tuple("TERMINAL_EXITS")

dec=next(n for n in ast.walk(cls) if isinstance(n,ast.FunctionDef) and n.name=="_decide_once")

emitted_a, emitted_e = set(), set()
plans=[]
for n in ast.walk(dec):
    if isinstance(n,ast.Call) and getattr(n.func,"id","")=="plan":
        if n.args:
            for c in ast.walk(n.args[0]):
                if isinstance(c,ast.Constant) and isinstance(c.value,str):
                    emitted_a.add(c.value)
        a=n.args[0].value if n.args and isinstance(n.args[0],ast.Constant) else "<expr>"
        for kw in n.keywords:
            if kw.arg=="exit_":
                for c in ast.walk(kw.value):
                    if isinstance(c,ast.Constant) and isinstance(c.value,str): emitted_e.add(c.value)
        plans.append((n.lineno,a))
# plan() 内部的改写也会产生动作/出口
for n in ast.walk(dec):
    if isinstance(n,ast.Assign):
        tnames=set()
        for t in n.targets:
            for c in ast.walk(t):
                if isinstance(c,ast.Name): tnames.add(c.id)
        if {"action","exit_"} & tnames:
            for c in ast.walk(n.value):
                if isinstance(c,ast.Constant) and isinstance(c.value,str):
                    (emitted_a if c.value in ACTIONS else emitted_e).add(c.value)

print("="*68); print("1) 死动作：声明了但 decide() 发不出")
dead_a=[a for a in ACTIONS if a not in emitted_a]
for a in dead_a: print(f"   ✗ {a}")
print(f"   （声明 {len(ACTIONS)}，发得出 {len(ACTIONS)-len(dead_a)}）" if dead_a else "   （无）")

print("\n2) 死出口：声明了但 decide() 发不出")
dead_e=[e for e in EXITS if e not in emitted_e]
for e in dead_e: print(f"   ✗ {e}")
if not dead_e: print("   （无）")

print("\n3) 执行器认不认识每个动作（不认识 ⟹ HALTED_NO_EXECUTOR）")
disp=set()
for n in ast.walk(ptree):
    if isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id in ("act","action","_act"):
        for c in n.comparators:
            for x in ast.walk(c):
                if isinstance(x,ast.Constant) and isinstance(x.value,str): disp.add(x.value)
no_exec=[a for a in ACTIONS if a not in disp and a not in ("DONE","NO_ACTION")]
for a in no_exec: print(f"   ✗ {a}  执行器没有分支")
if not no_exec: print("   （全部有执行器）")

print("\n4) 赋值了但从未被读的局部量（死计算）")
assigned=collections.defaultdict(list); loaded=set()
for n in ast.walk(dec):
    if isinstance(n,ast.Name):
        if isinstance(n.ctx,ast.Store): assigned[n.id].append(n.lineno)
        else: loaded.add(n.id)
dead_v=[(v,ls) for v,ls in assigned.items() if v not in loaded and not v.startswith("__")]
for v,ls in sorted(dead_v, key=lambda x:x[1][0]):
    print(f"   ✗ {v:32s} 赋值于行 {ls}")
if not dead_v: print("   （无）")

print("\n5) 同一个谓词被测两次（可能后一次恒假 = 死支）")
sig=collections.defaultdict(list)
for n in ast.walk(dec):
    if isinstance(n,ast.If):
        try: sig[ast.unparse(n.test)].append(n.lineno)
        except Exception: pass
for t,ls in sig.items():
    if len(ls)>1: print(f"   ! 行 {ls}: {t[:88]}")

print("\n6) return 之后同层还有语句（不可达）")
def scan(body,where):
    for i,st in enumerate(body[:-1]):
        if isinstance(st,(ast.Return,ast.Raise,ast.Continue,ast.Break)):
            print(f"   ✗ {where} 行 {body[i+1].lineno} 起不可达（前一句是 "
                  f"{type(st).__name__} @ {st.lineno}）")
    for st in body:
        for f in ("body","orelse","finalbody"):
            if hasattr(st,f): scan(getattr(st,f), where)
scan(dec.body,"_decide_once")
print()
