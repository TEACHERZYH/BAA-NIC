# 最小实验复现流程

## 1. 环境

论文主实验使用 Python 3.9.18、PyTorch 2.8.0、CUDA 12.8 和 CompressAI
1.2.8。安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

不同 CUDA 平台可先按 PyTorch 官方方式安装匹配的 PyTorch，再安装其余依赖。

## 2. 数据

实验使用 VisDrone 和 UAVid。仓库保留冻结的划分与裁剪清单，但不分发数据集
图像。运行脚本时通过 `--data-root`、`DATA_ROOT` 或 `UAVID_ROOT` 指向数据根目录。
当提供 `data_root` 时，加载器根据清单中的 `source_id` 定位图像，原执行机器的
`source_path` 不参与路径解析。

## 3. 无数据验证

```bash
python -m compileall -q models scripts
python scripts/verify_published_results.py
python scripts/smoke_core_v3_controller.py
python scripts/smoke_core_v3_bandwidth.py
python scripts/audit_core_v3_experiment_plan.py
python scripts/audit_core_v3_decoder_extension.py
CORE_V3_SEED_EXPANSION_PATH=configs/core_v3_decoder_five_seed_addendum.json \
  python scripts/audit_core_v3_seed_expansion.py
```

PowerShell 中可先执行：

```powershell
$env:CORE_V3_SEED_EXPANSION_PATH = "configs/core_v3_decoder_five_seed_addendum.json"
python scripts/audit_core_v3_seed_expansion.py
```

## 4. 最小完整执行顺序

1. G0a：结构、实际熵码流、CRC、独立解码和累积前缀检查。
2. 三个联合训练种子：17、29、43。
3. G0b：最终检查点实际码流闭环。
4. E1：完整发送端复杂度与 dense control 对照。
5. E2：固定发送端缓存上的 light/heavy 解码器训练与同字节评估。
6. E4 接口验证：构建 16 动作缓存、OpenJPEG operational reference、因果轨迹和回放。
7. 增强解码器：在同一发送端缓存上完成 17、29、43、59、71 五种子评估。
8. 运行三个分析脚本和 `validate_core_v3_results.py`。

Slurm 入口均位于 `scripts/slurm_core_v3_*.sh`。各脚本使用可覆盖的
`PROJECT_ROOT`、`PYTHON`、`DATA_ROOT` 和 `UAVID_ROOT` 环境变量。

## 5. 证据边界

冻结实验支持实际可解码码流、完整发送端轻量化、同字节服务器质量补偿和
bandwidth-operable 16 动作接口。固定物理组顺序的质量最优性以及控制器相对
best-fixed 或 capacity-greedy 的性能优越性不属于本仓库支持的论文主张。
