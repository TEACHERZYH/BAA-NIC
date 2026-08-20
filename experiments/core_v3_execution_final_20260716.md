# Core-v3 最终执行与主张审计（2026-07-16）

## 冻结合同

- 计划：`experiments/minimum_sufficient_experiment_plan_literature_aligned_20260715.md`
- 合同：`configs/core_v3_experiment_contract.json`（`core_v3_20260715_r3`）
- 合同 SHA256：`82de33fe6659bd741d774251e01e7602e4460a32770790dc336b7a7ff199b26c`
- 正式种子：`17/29/43`

## 证据审计

| Gate | 结果 | 可支持的论文表述 |
|---|---|---|
| C0/G0b | PASS | 每个独立 prefix 均由实际 entropy coder 生成、解码并通过确定性 round-trip；主结果不依赖 rate proxy。 |
| G1/E1 | PASS | UAV 发送端 analysis transform 在参数量与 MACs 上显著轻于冻结 anchor；完整发送路径计入 entropy coding 与 packing。 |
| G2/E2 | PASS | 在 byte-identical bitstream 条件下，服务器 heavy decoder 相对 light decoder 在 VisDrone 与 UAVid 上均产生稳定且显著的 MS-SSIM 增益。 |
| G3/E3 | FAIL | 只能报告 prefix 字节严格递增和接口可独立解码；不能声称冻结组顺序在质量或预算效用上最优。 |
| G4/E4 | FAIL | 只能报告控制器为 causal、可运行且满足冻结非劣条件；不能声称其优于 best-fixed 或 capacity-greedy。 |

## 规范产物

- 统计审计：`experiments/core_v3_analysis/quality_gate_audit.json`、`experiments/core_v3_analysis/bandwidth_gate_audit.json`
- 主表：`tables/core_v3_lightweight_asymmetry.csv`、`tables/core_v3_decoder_asymmetry.csv`
- 本地证据包：`runs/core_v3_remote_sync/final_archives/core_v3_evidence_small.tar.zst`
- 超大明细归档：`runs/core_v3_remote_sync/final_archives/core_v3_bandwidth_detail.csv.zst`
- 资源与费用审计：`experiments/core_v3_resource_cost_audit_20260716.md`；
  连续墙钟估算为 `26.124 A100 GPU-h`、`255.269 CPU core-h` 和
  `133.17 CNY`（不含平台取整、存储及其他费用）。

## 最终判断

冻结的最小实验集合已全部执行并审查。论文应以 C0、G1 和 G2 为主证据；G3/G4 作为一次简洁的适用边界说明，不再驱动标题、摘要、贡献或结论。所有失败、重试和历史产物均保留。

## 稿件与投稿包闭环

- 英文/中文稿已按上述证据边界对齐，公式 (1)-(10) 与代码/合同审计通过。
- Remote Sensing 模板 DOCX 已重建并视觉检查为 15 页；Figure 1-2、
  Table 1-4、图注、公式编号和参考文献均无越界或重叠。
- Claim、formula、reference 和中文稿审计为 `PASS`；MDPI 与 submission
  readiness 为 `PASS_WITH_WARNINGS`，警告仅为最终投稿元数据占位符。
- 当前 review package：`BAA-NIC_core_v3_review_package_20260716.zip`，
  SHA256 `23a66d81df4c6d583e522d2cef3b1f785e5e0a576f9b8d6c718353b63f69bbd2`。
