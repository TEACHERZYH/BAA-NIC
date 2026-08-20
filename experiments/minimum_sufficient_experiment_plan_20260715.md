# BAA-NIC 最小充分实验计划（2026-07-15）

> 已由 `minimum_sufficient_experiment_plan_literature_aligned_20260715.md`
> 取代为当前执行合同。本文件仅保留用于版本追溯。

## 1. 计划定位

本文件记录被取代版本，仅用于追溯，不得作为后续实验入口。当前唯一执行合同见文件顶部指向的 literature-aligned 计划及其机器可读 JSON。

本计划不声称：

- 在 PSNR、MS-SSIM、LPIPS 或低码率率失真上超过强传统或强 learned codec；
- 达到 SOTA、官方 VisDrone 检测优势或小目标恢复优势；
- 已在 RK3588 等真实边缘设备上达到实时部署或能耗优势；
- 教师蒸馏、ROI side stream、生成式解码或 v2-v8 分支构成主贡献。

## 2. 冻结主张与证据

| 编号 | 论文主张 | 最小直接证据 | 不允许替代的证据 |
|---|---|---|---|
| C0 | BAA-NIC 是可解码的渐进式压缩系统 | 实际 `compress/decompress`、总字节数、100% round trip | target bpp、rate proxy、仅 likelihood bpp |
| C1 | UAV 端编码器轻量 | 编码器参数量、MACs/GFLOPs、峰值激活内存和独立编码延迟；与同通道 dense-conv 内部控制比较 | 全模型延迟冒充编码器延迟、跨硬件速度表 |
| C2 | 非对称设计把复杂度移到服务器，并在完全相同码流下改善重建 | 共享冻结编码器和 entropy model；重/轻解码器读取同一批 bitstream；复杂度拆分和配对质量差异 | 不同编码器、不同 bpp 或不同训练预算的比较 |
| C3 | 带宽控制器比固定策略更适合波动链路 | 实际码流字节驱动的 trace replay；adaptive 对同一模型四个 fixed-prefix 策略 | 把 oracle 选择结果命名为 controller、用预设 bpp 代替实际字节 |

论文主线只保留 C1-C3；C0 是所有压缩与链路实验的前置有效性门。

## 3. 冻结模型与数据范围

- 主模型入口：`models/baa_nic.py::BAANICModel`。只允许补齐实际量化、entropy coding 和 bitstream API，不切换到 v5/v7/v8、AEIC Track B 或外部 teacher 路线。
- 渐进表示：4 个有序 latent groups，对应 active ratio `0.25/0.50/0.75/1.00`。
- `q_index` 必须实际控制量化步长和最终字节数，不能只作为返回字段或策略标签。
- 主训练集：VisDrone train。
- 主测试集：VisDrone val 全部有效样本。
- 跨场景复核：UAVid val 全部有效样本，仅评估，不再训练第二套模型。
- 输入协议：训练使用确定性记录的 512 crop 规则；复杂度统一使用 `1x3x512x512`；重建评估使用冻结 manifest 和确定性预处理。
- 训练随机种子：`17/29/43`。所有可训练主模型结果报告 3 个种子的均值和标准差。
- DOTA、xView、任务检测和 ROI 结果保留在历史或补充材料，不属于本计划验收门。

## 4. P0：实际码流闭环

### 实现要求

1. 为每个 active latent group 实现实际量化和 entropy coding。
2. `compress()` 输出头信息和 group payload；头信息至少包含图像尺寸、active groups、`q_index`、各流长度和模型版本。
3. `decompress()` 只依赖 bitstream 和 checkpoint 恢复重建，不读取发送端 latent。
4. 实际 bpp 按全部 payload 与 header 的总字节计算：`8 * total_bytes / (H * W)`。
5. 同一图像、同一 checkpoint、同一配置重复编码应产生可复现的解码结果。

### 最小验证

- VisDrone 与 UAVid 各 32 张图像；4 个 active ratios；全部有效 `q_index`。
- active ratio 增大时，总码流字节数必须单调不减。
- `q_index` 改变时，实际字节数和重建质量必须产生可测响应。
- round trip 成功率必须为 100%；失败时不得进入后续主实验。

### 输出

- `data/core_v2_bitstream_roundtrip.csv`
- `data/core_v2_operating_points.csv`
- `runs/core_v2_bitstream/metadata.json`

## 5. E1：轻量编码与算力拆分

### 对照

- `BAA-depthwise-encoder`：论文主编码器。
- `dense-conv-control`：保持 stage channels、下采样率、latent channels 和输入尺寸不变，只把 depthwise-separable block 替换为标准卷积；该行只用于结构复杂度对照，不做性能排名。
- `server-heavy-decoder` 与 `symmetric-light-decoder`：用于展示 encoder/decoder 复杂度分配。

### 指标与协议

- 参数量、MACs/GFLOPs、峰值激活内存；分别统计 encoder、entropy model、decoder 和 full model。
- batch size 1、FP32、512 x 512；50 次 warm-up，300 次测量；报告均值、标准差和 95% CI。
- 编码器、解码器和全模型分别计时，不再使用全模型延迟代替 UAV 编码延迟。
- 平台信息完整记录；不把同平台测试写成真实 UAV 设备结论。

### 支持门槛

- 主编码器相对 dense-conv control 的参数量和 GFLOPs 均至少降低 50%；
- 主模型的 server decoder 计算量显著高于 UAV encoder，形成明确的算力非对称；
- 若任一条件不满足，删除相应“轻量”或“算力转移”强表述。

### 输出

- `data/core_v2_complexity_split.csv`
- `tables/core_v2_lightweight_asymmetry.csv`

## 6. E2：同码流非对称解码器对照

### 训练设计

每个种子先使用临时 heavy decoder 完成一次主模型预训练，使 encoder 与 entropy model 收敛。随后冻结二者、丢弃临时 decoder，并缓存训练码流；最终比较的两个 decoder 均从头训练：

- `server-heavy-decoder`；
- `symmetric-light-decoder`。

两个 decoder 使用相同训练样本、损失、优化器、更新次数、初始化规则和早停规则。评估时二者读取逐字节相同的 bitstream，因此 bpp 完全一致，差异只来自接收端复杂度。

### 指标

- 4 个 active ratios 下的实际 bpp、PSNR、MS-SSIM 和 LPIPS；
- VisDrone 为主表，UAVid 为跨场景复核；
- 图像级 paired bootstrap 95% CI；种子级报告 mean +/- std。

### 支持门槛

- 重解码器在相同码流下至少一项主要重建指标稳定优于轻解码器，且 95% CI 不跨 0；
- encoder 侧参数量、GFLOPs 和 bitstream 字节数必须保持不变；
- 若未通过，只能声称“结构上分离编码与解码”，不能声称服务器复杂度带来有效补偿。

### 输出

- `data/core_v2_decoder_asymmetry.csv`
- `data/core_v2_decoder_asymmetry_per_image.csv`
- `tables/core_v2_same_bitstream_decoder.csv`

## 7. E3：实际字节驱动的带宽自适应

### 策略

- `BAA-controller`：调用代码中的真实 controller，根据可用 payload bits 选择 active groups 和 `q_index`。
- `fixed-r0.25/r0.50/r0.75/r1.00`：同一 checkpoint 的四个固定前缀策略，不需要额外训练。
- `best-fixed`：只在开发 traces 上选择一个全局固定策略，再冻结到测试 traces。
- `prefix-oracle`：逐帧事后最优上界，只作为灰色参考，不得命名为 controller 或主要 baseline。

### 链路与统计协议

- 三类独立测试 traces：平滑波动、突发降带宽/恢复、burst packet loss。
- 每类 30 个随机种子，每条 300 帧；开发 traces 与测试 traces 的种子和参数范围分离。
- 每帧可用 bits 由 bandwidth、deadline、packet loss 和协议开销共同计算；发送判定使用 P0 的实际总字节数。
- 报告 delivery rate、severe-link delivery、deadline violation、bitrate overshoot、all-frame PSNR/MS-SSIM（丢帧计 0）、delivered-frame quality 和 policy switch rate。
- 以 trace 为统计单位，报告 95% CI，并对 `BAA-controller` 与 `best-fixed` 做配对检验。

### 支持门槛

- controller 在至少两类测试 traces 上显著提高 all-frame utility 或 delivery rate；
- bitrate overshoot 和 deadline violation 不高于 best-fixed；
- 四级 active ratio 与 `q_index` 均被实际触发，且没有使用事后 oracle 信息；
- 若仅能展示策略切换而没有统计收益，论文只能声称“可运行的带宽响应”，不能声称优于固定策略。

### 输出

- `data/core_v2_bandwidth_trace_summary.csv`
- `data/core_v2_bandwidth_trace_detail.csv`
- `data/core_v2_bandwidth_statistics.csv`
- `figures/fig_core_v2_bandwidth_adaptation.*`

## 8. 最小实验矩阵与训练预算

| 分支 | 需要训练 | 种子 | 目的 |
|---|---:|---:|---|
| 主模型预训练（encoder + entropy model + 临时 decoder） | 是 | 3 | 生成冻结 encoder 和最终实际码流 |
| heavy decoder（冻结 encoder 后） | 是，仅 decoder | 3 | 非对称主结果 |
| light decoder（冻结 encoder 后） | 是，仅 decoder | 3 | 同码流因果对照 |
| dense-conv encoder control | 否 | - | 复杂度结构对照 |
| 4 个 fixed-prefix 策略 | 否 | - | 带宽策略对照 |
| controller trace replay | 否 | 30 traces/类型 | 自适应主结果 |

这已经是最小充分训练集合：3 个主训练加 6 个 decoder-only 训练。不得新增 no-ROI、detector、teacher、AEIC、MLIC++、LIC-TCM、Cheng2020、MBT2018 或生成式分支，除非用户再次明确变更计划。

## 9. 执行顺序与并行边界

1. 本地 `dl_env`：记录环境；完成 bitstream、单图 round trip、encoder/decoder 独立计时和 2-image trace smoke。
2. P0 通过后，检查新 Slurm 主机分区、队列和价格；先跑单 seed 短基准，记录吞吐、显存、预计墙钟和成本。
3. 三个主训练种子彼此独立，可在成本允许时并行提交；不得为形式顺序而串行。
4. 每个主种子通过后，并行启动对应 heavy/light decoder-only 训练。
5. 所有 decoder 分支完成后统一运行 E1、E2 和 E3；trace replay 只读取冻结 bitstream，不重新训练。
6. 每个分支按 checklist、日志、checkpoint、CSV schema 和验收门分别标记 `pass/blocked/failed`。
7. 远程训练每 30 分钟检查一次；完成后同步全部产物并执行账户级 Slurm/进程/同步/计费审计。
8. 仅在 G0-G4 全部审计后更新摘要、贡献、方法、主表和图；任务保持、ROI、教师模型和低码率性能比较退出主叙事。

## 10. 主文最小结果结构

- Table 1：模型各端参数量、GFLOPs、内存和同平台延迟。
- Table 2：同一 bitstream 下 heavy decoder 与 light decoder 的配对结果。
- Table 3：adaptive、best-fixed 和四个 fixed-prefix 策略的 trace 统计。
- Figure 1：UAV lightweight encoder / scalable bitstream / server-heavy decoder 系统图。
- Figure 2：实际 bpp 随 active ratio 和 `q_index` 的 operating-point 图。
- Figure 3：带宽、策略选择、实际 payload 和 delivered-frame utility 的时间序列。

主文不设置外部强模型性能排行榜。已有 BPG、VVenC、CompressAI、AEIC、MLIC++ 和 LIC-TCM 结果只作为补充背景或历史边界，不再新增训练，也不用于支撑 C1-C3。

## 11. 最终验收与论文回退规则

- Gate G0：实际 bitstream 100% round trip，实际字节计数有效。
- Gate G1：轻量编码器复杂度门通过，encoder-only 测量无误标。
- Gate G2：重解码器在相同字节下产生统计显著的重建补偿。
- Gate G3：真实 controller 在独立 traces 上优于 best-fixed，且无更高超载/违约率。
- Gate G4：所有主表均来自版本化 checkpoint、冻结 manifest 和 schema-valid CSV；3 seeds 与置信区间完整。

只有 G0-G4 全部通过，论文才保留“轻量化、非对称、带宽自适应”三项主张。任一门失败时按对应章节降级主张，不用新增无关实验补数量。历史 proxy-stage、ROI、task 和 teacher 数据保持原样，不覆盖 `data/real_*.csv`。
