# Core-v3 解码端扩展计划审查

- 版本：`core_v3_decoder_extension_20260717_r1`
- 父合同：`core_v3_20260715_r3`
- 审查结论：`PASS`

## 审查项

| 项目 | 结论 | 说明 |
|---|---|---|
| 主张一致性 | PASS | 只检验服务器端重建增强，不扩大轻量发送端或控制器主张。 |
| 因果隔离 | PASS | enhanced 与 heavy 消费同一 exact-FP32 熵解码缓存和同一 prefix bytes。 |
| 数据泄漏 | PASS | development 用于选择，VisDrone/UAVid test 只作一次正式评估。 |
| 最小充分性 | PASS | 一个增强候选、一个既有对照、三种子和两个冻结测试集。 |
| 统计完整性 | PASS | 配对层级 bootstrap、95% CI、Holm 校正和种子方向检查均已冻结。 |
| 失败回退 | PASS | pilot 或正式 gate 失败均停止扩展并保留结果，不修改原 G2。 |
| 费用合理性 | PASS | 先做 3090 benchmark，再按实测吞吐决定 3090/A100。 |

本审查只批准确定性同码流分支。扩散、生成式先验和额外侧信息不在本轮范围内。
