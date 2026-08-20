# Core-v3 实验计划最终审查记录

审查对象：

- `experiments/minimum_sufficient_experiment_plan_literature_aligned_20260715.md`
- `configs/core_v3_experiment_contract.json`
- `scripts/audit_core_v3_experiment_plan.py`

审查日期：2026-07-15
合同版本：`core_v3_20260715_r3`
结论：**PASS（计划与静态合同通过；尚未代表任何实验 gate 已通过）**

## 主张—证据闭环

| 主张 | 主对照 | 预注册主终点 | Gate | 未通过时的边界 |
|---|---|---|---|---|
| C0 实际压缩闭环 | 同 checkpoint 独立 round trip | actual bytes、独立解码、byte-prefix equality | G0a/G0b | 不进入论文主结果 |
| C1 发送端架构轻量 | dense control；CompressAI hyperprior 锚点 | 完整发送端参数与 MACs，时延作同平台佐证 | G1 | 仅保留相对 dense control 的降低事实 |
| C2 非对称解码收益 | symmetric light decoder | 同一 bitstream 的配对 MS-SSIM | G2 | 仅声称结构分离 |
| C3a 有序渐进表示 | reverse 和其余 22 种顺序 | 按实际字节归一化的 progressive AUC | G3 | 仅声称离散多档率控制 |
| C3b 带宽自适应 | best-fixed、capacity-greedy | `U_frame=delivered*MS-SSIM` | G4 | 仅声称可响应带宽 |

## 审查通过项

- 模型、q 语义、16 个训练 action、实际 entropy bytes 和可截断 bitstream 已冻结。
- development、VisDrone test 和 UAVid test 的图像来源、用途与 hash 隔离明确。
- heavy/light 结构在 G0a 前冻结；只统计有效执行路径，不能用无效模块填充复杂度。
- controller 不可读取图像、profile label、测试质量、未来链路或测试统计；无可行动作也有确定性 fallback。
- E4 明确 packetization、三阶段 FCFS 队列、因果 worker 状态、action-specific latency cache 和零预算定义。
- C2/C3/C4 的主终点、层级配对 bootstrap、Holm 校正、CI gate 和失败降级均已预注册。
- 主张范围限于实际 bitstream、同平台复杂度/时延和受控 stochastic trace replay，不外推实机或真实无线场测。
- 旧计划入口已标记为历史、不可执行，不得覆盖当前合同。

## 非阻塞剩余风险

- 计划通过不保证 G1-G4 的实验结果必然为正；负结果必须按预注册降级规则处理。
- CompressAI 锚点若无法满足 10% bpp 匹配容差，必须报告最近档位及差异，不能外推 RD 排名。
- 在 G0a 前仍需实现并冻结 CSV schema；G0a 后不得根据结果修改合同。

## 静态验证

- JSON parse：PASS。
- Python `py_compile`：PASS。
- `scripts/audit_core_v3_experiment_plan.py`：PASS。
- `git diff --check`：PASS（仅有工作区行尾转换提示，无 whitespace error）。

因此，当前计划可进入实现与 G0a 阶段；本审查未启动训练或修改模型实现。
