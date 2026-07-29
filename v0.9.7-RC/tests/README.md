# 测试

本目录包含全部 pytest 测试、共享 `conftest.py` 和固定离线测试入口。

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
