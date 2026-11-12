# BAA-NIC



BAA-NIC 面向发送端资源受限、服务器算力充足且无线带宽持续变化的遥感传输
场景。核心设计由轻量发送端、同字节服务器质量补偿和可按带宽选择的分层实际
码流组成。

## 核心证据

| 论文主张 | 公开证据 |
|---|---|
| 实际码流可用 | 11,520 条实际 entropy bitstream 记录全部通过独立解码和 CRC；3,072 条累积前缀保持 byte identity |
| 完整发送端轻量化 | 完整发送端为 0.635 M 参数、2.640 GFLOPs；analysis transform 相对 dense control 减少 53.16% 参数和 70.73% MACs |
| 服务器侧质量补偿 | 完全相同的发送字节和潜变量下，heavy decoder 在 VisDrone/UAVid 上分别提高约 0.0312/0.0411 MS-SSIM |
| 解码能力可继续扩展 | 五个独立完整流程种子中，enhanced decoder 在 VisDrone 四个前缀上继续提高约 0.0309 至 0.0324 MS-SSIM |
| 带宽可操作码流 | 4 个量化等级乘 4 个累积前缀形成 16 个可执行动作；72.9M 帧因果回放通过预设运行安全边界 |

## 快速验证

以下命令不需要数据集、GPU 或模型检查点：

```bash
python -m compileall -q models scripts
python scripts/verify_published_results.py
```

预期输出：

```text
PASS: compact BAA-NIC evidence matches the frozen contracts and manuscript claim scope
```

## 仓库结构

- `models/`：轻量分析变换、组级超先验、实际码流、控制器和三档服务器解码器。
- `scripts/`：训练、缓存、实际编码、质量/复杂度/带宽评估、统计和 Slurm 入口。
- `configs/`：冻结 core-v3 合同、结果 schema、动作调度和五种子补充合同。
- `data/`：冻结的 VisDrone/UAVid 划分与裁剪清单，不包含数据集图像。
- `results/`：直接支撑论文主张的小型结果、逐图统计和 gate audit。
- `experiments/`：预注册最小实验计划、审查记录、完成记录和主张边界。
- `docs/reproduction.md`：完整复现顺序和数据配置说明。

## 复现范围

本公开集合聚焦论文的三个部署属性：完整轻量发送、byte-identical server
compensation 和 bandwidth-operable scalable coding。它不以低码率率失真排名、
固定物理组顺序最优性或控制策略相对固定/贪心策略的优越性作为论文主张。

模型检查点、公开数据集图像、SQLite 缓存及 72.9M 行逐帧回放未纳入 Git。
紧凑统计、原始产物行数和 SHA-256 均保存在 `results/` 中，并可由所附脚本重建。

详细流程见 [docs/reproduction.md](docs/reproduction.md)。

## 引用与许可

正式引用信息将在论文发表后补充。代码许可尚未指定；未经许可不得据此推定
再分发或商业使用授权。
