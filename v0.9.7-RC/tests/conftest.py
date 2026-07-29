"""pytest 共享配置。

只做一件事：把仓库根目录放进 `sys.path`。

在此之前，`import ibs_engine` / `import abfe_pipeline` 能成功纯粹是因为
"pytest 恰好从仓库根目录启动"（rootdir 被隐式加进 sys.path）。从别处调用
（`python -m pytest /path/to/repo/test_x.py`、IDE 里点单个测试、或将来引入
CI 时换了工作目录）就会 ImportError。显式加进来把这个隐式依赖去掉。

刻意**不**在这里 import openmm / ibs_engine：
  - 在这台机上首次 `import openmm` 要 60-100 s（NFS 忙时更久），放进 conftest
    会让每次收集测试都先付这个代价，连 `--collect-only` 也逃不掉。
  - ATT-04 正在追踪"导入期 CUDA 初始化 / spawn 安全"的问题，conftest 是所有
    测试进程和 xdist worker 都必经的地方，不该在这里引入任何设备初始化副作用。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
