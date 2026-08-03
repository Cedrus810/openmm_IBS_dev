# 测试

本目录包含全部 pytest 测试、共享 `conftest.py` 和固定离线测试入口。

⚠️ **pytest 配置在仓库根目录的 `pytest.ini`，不在本目录。** 2026-07-30 之前它放在
`tests/pytest.ini`，而正式入口 `run_offline_tests.sh` 是从仓库根目录不带路径参数跑
pytest——pytest 只会从 cwd **向上**找配置文件，所以那份 ini 对全量 pre-flight 整体
静默失效（marker 未注册、`-ra` 未生效）。请勿把它移回本目录，也不要在本目录再放一份
副本（两份并存会把 rootdir 定成 `tests/`）。理由详见该文件顶部注释。

从仓库根目录运行：

```bash
./tests/run_offline_tests.sh
```

追加 pytest 参数或指定文件：

```bash
./tests/run_offline_tests.sh -x -q
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
```

测试若确实需要 CUDA，请标记为 `needs_gpu`；默认离线入口会排除这类测试。
普通物理、数值、协议和源码契约测试应保持 CPU 可运行。
