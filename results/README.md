# 结果证据说明

本目录仅保留能够直接核对论文核心主张的紧凑结果和统计审计。运行：

```bash
python scripts/verify_published_results.py
```

即可检查合同哈希、实际码流闭环、16 动作覆盖、完整发送端复杂度、同字节
解码器效应、五种子增强解码器结果以及带宽回放的证据边界。

## 已纳入的结果

- `core_v3_bitstream_roundtrip.csv`：11,520 条独立解码和 CRC 记录。
- `core_v3_operating_points.csv`：2 个数据集、3 个种子和 16 个动作的实际字节率。
- `core_v3_lightweight_asymmetry.csv`：发送端和解码端复杂度分解。
- `core_v3_decoder_asymmetry.csv`：heavy-minus-light 的配对统计。
- `core_v3_decoder_extension_summary.csv`：五种子 enhanced-minus-heavy 汇总。
- `core_v3_decoder_extension_per_image.csv`：五种子逐图效应。
- `core_v3_bandwidth_statistics.csv`：因果回放统计检验。
- 三个 `*_gate_audit.json`：冻结门槛与主张边界。

## 未纳入 Git 的大型派生产物

这些文件可以由仓库脚本重新生成，其行数和 SHA-256 保留如下：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `core_v3_complexity_breakdown.csv` | 93,000 | `f1cce0d078f887acc78ef28be49b852198ff6a5bbd750b392035583bd8646e90` |
| `core_v3_decoder_asymmetry.csv`（逐样本原始表） | 73,728 | `cbe49e7d01842c1737f61056454ac9bb52c37066bdb302dd29bf9faf41937e22` |
| `core_v3_progressive_ordering.csv` | 884,736 | `625fa0f4b269bd31763312139557bb00d9c73557474fa0df3868c9280929c349` |
| `core_v3_bandwidth_summary.csv` | 243,000 | `671f9e81b2540b21d82b7b55bfc25e646f5342e8e7c748f129f0cd5078b648b4` |
| `core_v3_bandwidth_detail.csv` | 72,900,000 | `4b049e27e784d8d04ff4a6bdbed5c5532ca0886001a0b6f646be6ad8efb1e4fd` |

模型检查点、SQLite 缓存、数据集图像和逐帧回放不进入版本库。
