# Core-v3 确定性重解码扩展计划（r1）

## 1. 继承边界

本扩展继承并保持 `core_v3_20260715_r3` 的发送端、组超先验、实际熵码流、
四个量化索引、四个累积前缀、三种子、冻结 manifests 和 exact-FP32
熵解码缓存。原合同及 C0/G1/G2/G3/G4 结果不被覆盖。

唯一新增变量是服务器解码器。增强分支由 `q_index`/已接收组数条件化潜变量
校正、Restormer 风格通道注意力、多尺度合成和置信度门控残差头组成。条件均
可由已有码流头与分段状态得到，不增加传输字节。

## 2. 最小证据链

| 阶段 | 数据 | 比较 | 完成条件 |
|---|---|---|---|
| D0 本地 smoke | 合成张量、最小缓存 | enhanced forward/backward | 确定性、有限梯度、尺寸与范围通过 |
| D1 远程 benchmark | seed 17 缓存 | 500 steps | 无 OOM/NaN，记录吞吐、显存和费用 |
| D2 development pilot | VisDrone development | enhanced seed 17 | 5000 steps 的 mean MS-SSIM 不低于 0.81 |
| D3 正式训练 | 冻结 train/development | enhanced seeds 17/29/43 | 每种子最多 30000 steps，按 development MS-SSIM 选 best |
| D4 确证评估 | 512 VisDrone + 256 UAVid tiles | enhanced vs 原 heavy | 同一缓存、同一字节、三种子配对统计 |

VisDrone 四个 prefix 的主效应是在每个图像和种子内对四个 `q_index` 取均值后，
计算 enhanced-heavy 的 MS-SSIM 配对差。使用 10000 次层级 bootstrap 和 Holm
校正；至少 3/4 prefix 的 95% CI 下界大于 0、校正后 `p<0.05`，且对应三个
种子点估计均非负，才允许形成新增确证性主张。UAVid 只验证方向一致性。

## 3. 训练与选择约束

- 结构和损失由 `configs/core_v3_decoder_extension_contract.json` 冻结。
- development 只用于 checkpoint 选择和 D2 pilot gate；测试集不参与选择。
- 原 heavy checkpoint 是正式对照，不重训、不改损失、不改历史结果。
- D2 未通过时停止扩展并保留负面结果，不基于测试集调参。
- D4 未通过时不替换论文现有 G2，只作为解码端扩展负面证据。

## 4. 资源边界

先在 `3090` 做 500-step benchmark；若显存不足或预计总时长超过 A100 的
约 3.85 倍，则转 A100。按固定单价，A100 为 5.00 元/卡时，3090 为
1.30 元/卡时，CPU 为 0.01 元/核时。每次提交记录资源、预计上限、实际
卡时和取消/迁移原因。任何阶段结束后执行账户级 Slurm 与残留进程审计。
