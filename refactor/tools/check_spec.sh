#!/bin/bash
# spec 等价验收：引擎产物 vs 历史产物。
#   ① 行尾归一后逐字节相同（证明连注释空行都没动）
#   ② AST 等价（证明逻辑一致）
# 不用裸 cmp：历史上有几刀把 CRLF 写进纯 LF 文件，产物混合行尾，那是缺陷不复刻。
set -u
spec=$1; before=$2; after=$3; sub=$4      # sub = 目标文件在 root 下的相对路径
root=$(mktemp -d); mkdir -p "$root/$(dirname "$sub")"
cp "refactor/work/$before" "$root/$sub"
if ! PYTHONPATH=refactor/tools python3 refactor/tools/cutter.py "refactor/cuts/specs/$spec.py" --root "$root" >/tmp/cs.$$ 2>&1; then
  echo "  ❌ $spec 执行失败"; sed 's/^/     /' /tmp/cs.$$ | head -3; rm -rf "$root" /tmp/cs.$$; exit 1
fi
a=$(sed 's/\r$//' "$root/$sub" | md5sum | cut -d' ' -f1)
b=$(sed 's/\r$//' "refactor/work/$after" | md5sum | cut -d' ' -f1)
ast_ok=$(python3 -c "
import ast,sys
def n(p):
    t=ast.parse(open(p,encoding='utf-8',errors='replace').read())
    for x in ast.walk(t):
        for k in ('lineno','col_offset','end_lineno','end_col_offset'):
            if hasattr(x,k): setattr(x,k,None)
    return ast.dump(t)
print('Y' if n('$root/$sub')==n('refactor/work/$after') else 'N')")
rm -rf "$root" /tmp/cs.$$
if [ "$a" = "$b" ] && [ "$ast_ok" = "Y" ]; then echo "  ✅ $spec  行尾归一后逐字节相同 + AST 等价"; exit 0; fi
echo "  ❌ $spec  字节$([ "$a" = "$b" ] && echo ✅ || echo ❌)  AST$([ "$ast_ok" = Y ] && echo ✅ || echo ❌)"; exit 1
