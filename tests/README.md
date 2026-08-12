# 测试与最低验证

[项目入口](../README.md) · [维护指南](../docs/MAINTAINING.md)

本目录包含 pytest 测试、共享 `conftest.py` 和固定离线测试入口。测试覆盖物理数值、协议身份、
缓存与 resume、源码契约和实验模块，但测试通过不自动等于某一新科学结果已验证。

## 标准入口

从仓库根目录运行：

```bash
./tests/run_offline_tests.sh
```

追加 pytest 参数或指定文件：

```bash
./tests/run_offline_tests.sh -x -q
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
```

默认入口排除标记为 `needs_gpu` 的测试。真正依赖 CUDA 的测试必须显式使用该 marker。

## pytest 配置位置

`pytest.ini` 必须位于仓库根目录。不要在 `tests/` 再放一份副本，否则 pytest 的 rootdir、
marker 注册和默认参数可能发生静默变化。

## 测试层次

- 单元/数值测试：公式、符号、极限和小型合成数据；
- 协议契约测试：版本、lambda 路径、缓存身份和 fail-closed 行为；
- resume/状态测试：checkpoint、不可变 production 和恢复边界；
- CPU pre-flight：代码修改后的最低门；
- GPU/真实体系验证：环境、性能和完整 pipeline 的额外证据；
- 独立重复：科学结果的必要证据，不由普通 pytest 替代。

## 新增测试

- 优先使用小型、确定、CPU 可运行的 fixture。
- 对随机过程固定 seed，同时测试 seed 是否进入 provenance。
- 对 bug 同时覆盖失败条件和正确修复。
- 不读取或改写受保护的 production 输出；必要时使用临时目录或只读 fixture。

