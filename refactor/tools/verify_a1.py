import ast,collections,pathlib,sys
OLD=pathlib.Path('tests'); NEW=pathlib.Path(sys.argv[1])/'tests'
def asserts(tree):
    return [n for n in ast.walk(tree) if isinstance(n,ast.Assert)]
def lits(node):
    return sorted(n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str))
def chained(tree):
    out=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Compare) and len(n.ops)>1 and any(isinstance(o,(ast.In,ast.NotIn)) for o in n.ops):
            out.append(getattr(n,'lineno','?'))
    return out
bad=collections.Counter(); rep=[]
changed=[p for p in sorted(NEW.glob('test_*.py')) if (OLD/p.name).exists() and (OLD/p.name).read_text(encoding='utf-8',errors='replace')!=p.read_text(encoding='utf-8',errors='replace')]
print(f"被改写的测试文件: {len(changed)}")
for p in changed:
    o=(OLD/p.name).read_text(encoding='utf-8'); n=p.read_text(encoding='utf-8')
    try: to=ast.parse(o); tn=ast.parse(n)
    except SyntaxError as e:
        rep.append((p.name,'SYNTAX',str(e))); bad['syntax']+=1; continue
    ao,an=asserts(to),asserts(tn)
    if len(ao)!=len(an):
        rep.append((p.name,'ASSERT-COUNT',f'{len(ao)} -> {len(an)}')); bad['count']+=1
    lo=sorted(l for a in ao for l in lits(a)); ln=sorted(l for a in an for l in lits(a))
    if lo!=ln:
        add=collections.Counter(ln)-collections.Counter(lo); rm=collections.Counter(lo)-collections.Counter(ln)
        rep.append((p.name,'LITERAL-DRIFT',f'+{list(add)[:3]} -{list(rm)[:3]}')); bad['literal']+=1
    co,cn=chained(to),chained(tn)
    if len(cn)>len(co):
        rep.append((p.name,'CHAINED-COMPARE',f'新增 {len(cn)-len(co)} 处 @ {cn[:5]}')); bad['chained']+=1
    for kw in ('skip','xfail'):
        if n.count(kw)>o.count(kw): rep.append((p.name,'SKIP/XFAIL',kw)); bad['skip']+=1
print("问题统计:",dict(bad) or "无")
if rep:
    print("\n明细:")
    for f,k,d in rep: print(f"  [{k:16s}] {f}\n      {d}")
