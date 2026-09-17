# 出厂 R1 冻结权重（Atenolol）—— 2026-09-17 移出 `resources/`

绑的是 **Atenolol（41 原子，指纹 `99564d84b2cbe765…`，训练 EXP-020）**。
Atenolol 这个体系早已收工，于是**任何当前配体都指纹对不上**，
`--outer-lambda-local-residual-ibs` 一开就落进 EXP-033 P1 自动重训。

移出的理由：这份权重对现在的任何工作都不可用，留在 `resources/` 只会
让每条新配体的 run 都绕一次"加载冻结 → 指纹不匹配 → 自动重训"。
2026-08-31 本来就已经决定过 residual sampling 不随首发。

要用冻结权重：`--outer-lambda-resource-manifest <path>` 显式指过去
（身份门一道不放宽：配体指纹、原子数、payload/weights sha、插件源码 sha 照常校验）。
重训一份见 `docs/RETRAIN_LOCAL_RESIDUAL.md`。
