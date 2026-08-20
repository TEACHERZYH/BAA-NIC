# BAA-NIC 文献对齐的最小充分实验计划（2026-07-15）

## 1. 计划定位

本计划参考同主题高水平论文的实验方法，重新定义 BAA-NIC 的最小主实验集合。论文只验证三项主张：

1. 面向 UAV 侧的发送端架构在参数量、MACs 和同平台时延口径下是轻量的；
2. 复杂度由发送端转移到服务器端，并由重解码器产生可测收益；
3. 对每帧在编码前确定一个量化档后，仅生成一次渐进码流；该码流能够根据动态带宽选择可解码前缀，并优于同一模型的固定策略。

本计划不进行强模型性能竞赛，不声称低码率率失真优势、SOTA、检测优势、真实边缘设备实时性或能耗优势；C1 的“轻量”仅指冻结协议下的模型与计算代价，不等同于实机部署结论。既有外部 teacher、AEIC、MLIC++、LIC-TCM、Cheng2020、MBT2018、ROI、检测和生成式分支保留为历史证据，不进入新主实验。

## 2. 文献方法依据

| 文献 | 可复用的实验方法 | 在 BAA-NIC 中的采用方式 |
|---|---|---|
| Reduced-Complexity VAE, Remote Sensing 2021 | 将 onboard coding complexity 与 ground decoding 分开，报告参数和 FLOPs/pixel，并在多类遥感场景验证 | 分别统计 analysis transform、hyper/entropy coding、packing、controller 和 server decoder |
| ELIC, CVPR 2022；ShiftLIC, IEEE TCSVT 2025 | 同时报告压缩性能、参数量、KMACs/pixel、GPU/CPU encoding/decoding latency；消融使用接近 bpp | 使用实际字节，发送端/接收端分开计时；重/轻 decoder 使用逐字节相同码流 |
| DPICT, CVPR 2022；ProgDTD, CVPRW 2023 | 验证单一可截断码流的多个重建点，并比较正常、无优先级和反向优先级 | 对 4 个 latent groups 穷举 24 种无重编码传输顺序，并把正常顺序与 reverse/alternative distribution 比较 |
| Bandwidth-Agile DeepJSCC, IEEE TWC 2021 | 单模型覆盖多个带宽层，逐层报告重建，并与固定层策略比较 | controller 与同一 checkpoint 的 fixed-prefix 及简单 capacity-greedy 策略比较，不与强 codec 排名 |
| Progressive LIC over wireless channels, npj Wireless Technology 2026 | 在相同信道条件下比较策略，使用大量独立信道实现，报告平均/尾部等待时间、吞吐和质量 | 预计算 codec operating points，再进行 1000 次独立 goodput/packet-loss replay，报告 p50/p95/p99/p99.9 |
| RS-LLIC, IEEE TGRS 2026 | 采用 onboard encoding / ground decoding 范式，并在多个遥感数据集验证 | VisDrone 为主数据集，UAVid 为冻结的跨场景复核集 |

这些文献只用于确定实验规范。BAA-NIC 不复现其主模型，也不把与它们的 RD 排名设为验收条件。

## 3. 冻结主张和最小证据链

| 主张 | 必须回答的问题 | 最小证据 |
|---|---|---|
| C0：实际压缩有效 | 是否存在可独立解码、可计数字节的码流？ | `compress/decompress`、总字节、100% round trip、硬量化 |
| C1：发送端架构轻量 | 完整发送端而非单个网络层的代价是多少？ | transform、entropy、packing、controller 的参数、MACs/pixel、内存和同平台 latency 分解 |
| C2：计算非对称有效 | 更重的服务器 decoder 是否在相同码流下改善重建？ | 共享冻结 encoder/entropy；重/轻 decoder 解码相同 bitstream；配对统计 |
| C3：渐进与带宽自适应有效 | 前缀是否按重要性排序，controller 是否优于固定策略？ | 前缀/顺序实验；独立动态链路 replay；fixed-prefix 和 best-fixed 对照 |

任何 proxy bpp、target bpp、全模型延迟或事后 oracle 都不能替代上述证据。

### 3.1. 冻结基线—指标矩阵

| 主张 | 必选实验基线/对照 | 对照回答的问题 | 主判定指标 |
|---|---|---|---|
| C0：实际压缩有效 | 无外部基线；同一 checkpoint 的独立 `compress/decompress` 闭环 | 输出是否为可计数字节、可独立解码的实际码流 | 实际 bytes/bpp、round-trip 成功率、重复编码一致性、截断可解码率 |
| C1：发送端架构轻量 | `dense-conv-control`（主因果对照）；CompressAI `bmshj2018-hyperprior`（可复现外部 learned-codec 锚点） | depthwise 设计是否确实降低代价；完整发送端是否处于标准 LIC 的轻量区间 | 完整发送端参数、checkpoint bytes、MACs/pixel、GFLOPs@512、peak memory、分项及总编码 latency |
| C2：计算非对称有效 | `symmetric-light-decoder` | 在发送端和逐字节码流完全不变时，服务器重解码器是否产生质量收益 | byte equality、PSNR、MS-SSIM、LPIPS、decoder 参数/MACs/latency、配对 95% CI |
| C3a：有序渐进表示有效 | `reverse-order`、其余 22 种 alternative orders | 正常前缀顺序是否优于非重要性顺序 | 实际字节下的质量轨迹、共同字节预算质量、progressive AUC、monotonic-violation rate、24-order rank |
| C3b：带宽自适应有效 | `best-fixed`（主静态基线）；`capacity-greedy`（简单自适应基线）；`fixed-r0.25/r1.00`（极端失效模式）；progressive OpenJPEG/JPEG2000（外部 operational reference） | controller 的收益是否超过固定策略和只按瞬时容量选最大可行前缀的简单规则 | delivery、all-frame utility、overshoot、deadline violation、waiting-time p95/p99、throughput、配对 trace 统计 |

基线选择遵循以下冻结规则：

1. CompressAI hyperprior 只提供完整发送端的可复现实验锚点。其 quality level 在 development tiles 上一次性选择为最接近 BAA-NIC `(q_index=1, full-prefix)` 中位实际 bpp 的档位；预注册匹配容差为相对 bpp 差异不超过 10%。若所有档位均超出容差，则使用最近档位并显式报告 bpp 差异，不外推 RD 排名。
2. ShiftLIC 作为 2025 年轻量 LIC 的近期文献参照，用于解释参数—计算—时延报告规范。其官方仓库当前未提供预训练权重，因此不进入必须复现的数值基线，也不以非官方 checkpoint 替代。
3. `best-fixed` 固定一个 development traces 上选定的 `(active_ratio, q_index)`；`capacity-greedy` 只使用当前 point-estimate payload、development utility/size calibration 和当前帧一次编码后的 segment lengths，不使用风险分位数、未来 trace 或测试质量。
4. `fixed-r0.50/r0.75` 参与 `best-fixed` 候选搜索并进入补充材料；主文只保留 fixed-low、fixed-high 和 best-fixed，避免重复行。
5. `prefix-oracle` 只作为灰色诊断上界，不属于基线，不进入显著性检验。Cheng2020、ELIC、MLIC++、LIC-TCM、AEIC 和 VVenC/BPG 不进入主实验矩阵，因为它们不能分别隔离 C1-C3 的因果问题，并会把论文重新导向非主张的 RD 排名。

## 4. 冻结模型、数据和统计协议

机器可读执行合同为 `configs/core_v3_experiment_contract.json`。所有 core-v3 数据、训练、bitstream、统计和 replay 入口必须加载该文件并把其 SHA256 写入 metadata；脚本默认值与本计划或 JSON 不一致时视为 gate failure。合同只允许在 G0a 前修订并递增 `contract_version`，G0a 后不得根据结果更改。

### 4.1. 模型与 practical entropy 合同

- 论文主入口保持为 `models/baa_nic.py::BAANICModel`，保留现有 `LightweightEncoder`、4 x 48-channel latent groups 和 `ServerDecoder` 的主体结构。
- 为完成主张证据，允许新增且仅新增：共享权重的逐组 practical hyperprior codec、真实量化步长、bitstream API、任意 group-subset mask、dense encoder control、symmetric light decoder 和风险感知 controller。旧 v2-v8、ROI、teacher、detector、refiner 和生成式分支不得接入该路线。
- 四个 latent groups 记为 `y_1,...,y_4`。每个 48-channel group 使用同一套共享权重的 group hyper-analysis/hyper-synthesis 和 entropy models 独立编码；不得用其他 group 的未传输值作为概率上下文。
- `q_index` 固定为 `0/1/2/3`，对应主 latent 量化步长 `Delta_q = 2^(q/2)`，即 `1.0000/1.4142/2.0000/2.8284`；hyper-latent 使用单位量化步长。group hyper-analysis 输入为 `abs(y_g/Delta_q)`，hyper-synthesis 输出 scaled-symbol domain 的 mean/scale；entropy coding 使用 `y_g/Delta_q` 及同域 mean/scale，解码后再乘 `Delta_q` 恢复 decoder 输入。训练用均匀噪声近似，验证和 bitstream 使用硬 rounding。
- 训练对 4 个 `q_index` 和 4 个 prefixes 独立均匀采样，共 16 个 action；cross-rate consistency 只比较相同 `q_index` 下的当前 prefix 与 full prefix。
- 新主实验中 `enhancement_layers` 恒为 0，不产生、传输或评价 ROI/detail side stream。
- 单帧先由 controller 在编码前选择 `q_index`，随后 encoder 和 group entropy codec 各执行一次完整四组编码；获得实际 segment lengths 后，controller 可在传输前将 nominal prefix 向下截断，但不得更换 `q_index` 或再次编码。

### 4.2. 数据划分

- 先把 VisDrone official train 源图的相对路径规范为 POSIX `/` 分隔形式，按 UTF-8 bytes 升序得到稳定 source-ID 列表，再用 NumPy `PCG64(seed=20260715).permutation` 做 source-image-level shuffle；前 256 张作为 development images，其余作为训练集。冻结排序后列表、置换索引及其 SHA256；development images 不参加梯度更新。
- 训练：剩余 VisDrone train 源图，随机 512 x 512 crop。
- development：每张 development image 固定一个 512 x 512 tile，共 256 tiles，用于 checkpoint 选择、CompressAI quality 匹配、segment-size/utility calibration 和 controller 参数冻结。
- 主测试：从 VisDrone official val 构建 512 个确定性 512 x 512 tiles；每个源图最多一个 tile。
- 跨场景复核：从 UAVid val 构建 256 个确定性 512 x 512 tiles；manifest 必须记录 sequence/clip ID。
- 所有 manifests 在训练前冻结，记录源图、split、坐标、padding 和 SHA256。development 仅可用于本节列明的 checkpoint 选择、外部锚点匹配、size/utility/latency calibration 和 controller 冻结，不得用于改变损失权重或已冻结结构；两个测试集不得参与任何选择或 calibration。

### 4.3. 统计总则

- 模型种子固定为 `17/29/43`，报告 seed mean +/- std。
- C2 的预注册主质量指标为 MS-SSIM；PSNR 和 LPIPS 为次要指标。C3a 的主指标为同一图像和 q 的共同实际字节预算、再按该 full stream bytes 归一化到 0-100% 后的 MS-SSIM progressive AUC。C3b 的主指标为 `U_frame = delivered * MS-SSIM`，未按 deadline 完成 base segment 时 `delivered=0`。
- 图像质量差异使用 10,000 次 hierarchical paired bootstrap：先重采样 seed，再重采样 VisDrone source image；UAVid 按 sequence/clip cluster 重采样，报告 95% CI。
- 动态链路以 realization 为配对单位，在每个 seed 内重采样 realization，再重采样 seed；所有主检验使用双侧 alpha=0.05，并对同一 gate 内的 profile/deadline 比较使用 Holm 校正。
- 所有 CI 同时报告 effect size；次要指标不用于替代未通过的主指标。

## 5. P0：实际渐进码流闭环

### 实现

1. 每个 group segment `S_g` 必须同时包含该组的 hyper-latent string 和 main-latent string；group codec 参数共享，但四个 segment 独立算术编码和独立解码。
2. 完整码流固定为 `H || S_1 || S_2 || S_3 || S_4`。固定长度 header `H` 的首 8 bytes 为 ASCII `BAANIC3` 加一个 `0x00` padding byte，并记录 format/model version、checkpoint hash、H/W、padding、`q_index`、四段长度、header CRC32 和 header length；每个 `S_g` 自带 group ID、segment length 和 segment CRC32。禁止使用覆盖 whole stream 的 checksum，header 不因截断而重写。
3. prefix-k 只能从完整码流在登记边界截断得到，因此其 `H || S_1...S_k` 必须与 full stream 的对应字节完全一致。ordered prefix 的 present group 数由文件长度和目录推断；任意 subset/alternative-order stream 由 segment mini-header 定位 group，缺失 group 在 decoder 输入中置零并设置显式 indicator。
4. `compress(x, q_index)` 对完整 latent 只运行一次 analysis transform，并对四组各编码一次；禁止试编码多个 `q_index`。`decompress(bitstream)` 只读取 checkpoint 和 bitstream，不访问发送端 tensor。
5. 总 bpp 包含 header、四组实际传输的 hyper/main strings、校验字段及其他必要 side information。likelihood bpp 只作诊断，论文结果使用实际文件字节。
6. 发送端路径包括 analysis transform、group hyper-analysis、发送端必需的 hyper-synthesis/probability prediction、entropy encode、量化缩放、packing 和 controller；共享参数只计一次，逐组运行时间全部计入。

### 验证

- VisDrone/UAVid 各 32 张，覆盖 4 个 prefixes、15 个非空 group subsets 和全部 4 个 `q_index`。
- full、prefix 和 subset round trip 成功率均为 100%；相同 checkpoint、输入、软件和设备上的重复编码字节完全一致。跨 CPU/GPU bitstream 一致性不作为本论文主张。
- 固定 `q_index` 时，prefix 增加必须使总字节单调不减，且 prefix 文件必须通过 byte-prefix equality 检查。
- 固定 prefix 时，development-set 中位实际 bpp 必须随 `q_index=0->3` 单调不增，并至少形成三个相对 bpp 差异大于 5% 的 operating points。
- 实际 bpp 与 likelihood bpp 同时记录，但论文主结果只使用实际 bpp。

### 输出

- `data/core_v3_bitstream_roundtrip.csv`
- `data/core_v3_operating_points.csv`
- `runs/core_v3_bitstream/metadata.json`

### Gate G0a/G0b

- `G0a structural smoke`：在本地 `dl_env` 使用初始化模型或 pilot checkpoint，对 VisDrone/UAVid 各 2 张图完成四档 q、四个 ordered prefixes 和 15 个 subsets 的 API/byte-prefix/CRC/independent-decode 检查。G0a 未通过时不得启动远程训练。
- `G0b evidence gate`：每个最终 selected checkpoint 均按上述 32+32 图协议复测。任一 full/prefix/subset 不能独立解码、byte-prefix equality 失败、`q_index` 不改变实际码流，或结果仍依赖 proxy/target bpp，则不得进入 E1-E4 和论文主结果。

## 6. E1：完整发送端轻量化与算力拆分

### 必选对照与外部锚点

- `BAA-depthwise`：论文主 encoder。
- `dense-conv-control`：stage channels、下采样、latent channels 和 entropy model 不变，仅将 depthwise-separable blocks 替换为标准卷积。该控制只评估复杂度，不训练、不做质量排名。
- CompressAI `bmshj2018-hyperprior`：使用官方预训练 MSE checkpoint；quality level 按第 3.1 节冻结，只在相同软硬件协议下提供完整发送端复杂度和时延锚点。
- `server-heavy-decoder`：最终接收端。
- `symmetric-light-decoder`：与发送端规模相近的接收端控制。

ShiftLIC 的论文数值只在相关工作或表下注明为跨论文上下文，不与本地实测 latency 混排，也不参与 G1 判定。

### 测量项目

- 完整发送端参数量和 checkpoint bytes，包含 analysis transform、共享 group hyperprior、entropy probability model 和 controller；
- BAA-NIC 编码复杂度按“一次 full four-group stream 生成”测量；不得因最终只发送低 prefix 而漏计未传输 group 的编码成本；
- MACs/pixel、GFLOPs@512；
- peak allocated memory 和 peak activation memory；
- transform、group hyper-analysis、发送端 hyper-synthesis/probability prediction、entropy encode、q scaling、packing、controller、entropy decode、接收端 probability prediction和 decoder 分项延迟；
- end-to-end encode/decode latency；
- 对 CompressAI 锚点额外记录实际 bpp、PSNR 和 MS-SSIM，确认比较处于相近工作区间；这些质量指标只作 operating-point guardrail；
- sender/full-codec 参数量和 MACs 比例只描述复杂度迁移，不作为轻量化 gate；
- CPU 与 GPU 分开测量，平台、线程数、精度、batch size 和库版本完整记录。

### 计时协议

- batch size 1、FP32、512 x 512；固定 100 张 VisDrone test tiles，每张重复 5 次，共 500 个正式样本；运行前执行 50 次不计入结果的 warm-up；
- CUDA 测量使用 event 并同步；CPU 固定线程数；
- 报告 mean、std、p50、p95 和 95% CI；
- BAA-NIC 与 CompressAI 锚点必须在相同设备、软件环境和输入清单内完成编码与解码；
- E4 使用的 OpenJPEG LRCP stream 在同一 100-image timing manifest 上测量 encode/decode latency，并写入相同 latency schema；
- 不把同平台 latency 写成 UAV 设备实测。

### Gate G1

- 主 analysis transform 相对 dense control 的参数量和 MACs 至少降低 50%；
- 完整发送端参数量和 MACs 应同时低于预注册 CompressAI hyperprior 锚点；未通过时，将“标准 LIC 参照下的轻量发送端”降级为“相对 dense control 的轻量化”；
- entropy coding 和 packing 必须计入发送端 latency；
- sender/full-codec 比例必须报告但不设阈值；
- 任一主 gate 未通过时，删除“标准 LIC 参照下的 lightweight sender”主张，只保留相对 dense control 的复杂度降低事实。

### 输出

- `data/core_v3_complexity_breakdown.csv`
- `data/core_v3_latency_samples.csv`
- `tables/core_v3_lightweight_asymmetry.csv`

## 7. E2：逐字节同码流的非对称解码器对照

### 训练设计

每个种子先用临时 heavy decoder 训练 encoder 和 entropy model。收敛后冻结发送端并丢弃临时 decoder；最终比较的 heavy/light decoders 均从头训练，使用相同数据、损失、优化器、训练预算、早停规则和随机初始化协议。`symmetric-light-decoder` 的 MACs/pixel 必须位于对应 analysis encoder 的 +/-10%，参数量必须位于 +/-20%；`server-heavy-decoder` 的参数量和 MACs/pixel 均须至少为 light decoder 的 2 倍。heavy/light 架构及其 hash 在 G0a 前冻结；参数量和 MACs 只统计实际参与至少一个 prefix 解码的执行路径，禁止加入未调用 block、恒等占位分支或其他不影响输出的参数来满足复杂度比。任何边界未满足都必须先改结构，禁止结果产生后放宽容差。

评估时先编码一次，再把完全相同的 bitstream 分别交给两个 decoders。不得分别重编码或通过 q 调整做近似 bpp matching。

### 指标

- 4 个 prefixes x 全部预注册 `q_index` 的实际 bpp、PSNR、MS-SSIM、LPIPS；
- decoder 参数、MACs/pixel、GPU/CPU latency；
- VisDrone 为主表，UAVid 为跨场景方向一致性检查；
- 选择 8 个固定图像，以同一裁剪位置提供 matched-bitstream qualitative crops。

主检验先在每张图像、每个 seed 内对四个 `q_index` 等权平均 MS-SSIM 差值，再按第 4.3 节做 hierarchical paired bootstrap。四个 prefix 检验使用 Holm 校正；PSNR、LPIPS 和单独 q 档位只作次要分解。

### Gate G2

- 两个 decoders 的 bitstream bytes 必须逐条完全相同；
- heavy decoder 在 VisDrone 至少 3/4 prefixes 上取得 Holm 校正后 MS-SSIM 改善且 95% CI 下界大于 0；相应三个 prefix 的三个 seed 点估计均不得为负；
- UAVid 聚合 MS-SSIM 差值不得为负；
- 未通过时，只能声称编码/解码复杂度结构分离，不能声称服务器重解码补偿有效。

### 输出

- `data/core_v3_decoder_asymmetry.csv`
- `data/core_v3_decoder_asymmetry_per_image.csv`
- `figures/fig_core_v3_same_bitstream_crops.*`

## 8. E3：渐进前缀与信息排序验证

该实验借鉴 DPICT/ProgDTD 的优先级消融，但不新增训练。

### 条件

- `ordered-prefix`：论文定义的 group 顺序；
- `reverse-order`：反向 group 顺序；
- `all-24-orderings`：穷举四个 group 的全部 24 种传输顺序，`ordered-prefix` 和 `reverse-order` 为其中两个预注册特例；
- 所有顺序均使用同一 checkpoint、同一图像、同一 `q_index` 和 P0 已编码的相同四个 group segments，不重新编码。非标准顺序通过重排独立 segment 的传输调度构造，不声称是原始 full stream 的物理 byte prefix。

### 评估

- 主评估使用全部 512 张冻结 VisDrone test tiles；256 张 UAVid test tiles 只作跨场景方向一致性检查，两个集合均不得用于顺序或阈值选择；
- 逐个 group 解码并绘制 actual-bpp/quality trajectory；
- 比较前缀增加后的图像级 PSNR、MS-SSIM 和 LPIPS 变化；
- 每张图像以完整四组码流总字节为 100%，在 25%/50%/75%/100% transmitted-byte budgets 上使用“最高已完整接收 segment subset”的真实重建；禁止按 group 数比较，也禁止插值生成不存在的重建图像；
- progressive AUC 对上述右连续阶梯质量函数在 0-100% byte budget 上积分，未完成 base segment 前质量记为 0；
- ordered-prefix 的单调容差固定为：MS-SSIM 下降超过 `1e-4`、PSNR 下降超过 `0.01 dB` 或 LPIPS 上升超过 `1e-4` 分别记一次 violation；
- 报告 ordered 顺序在全部 24 种顺序中的 AUC rank、相对 alternative median 的差值、reverse 差值和 monotonic-violation rate。

主检验先在每张图像和 seed 内对四个 `q_index` 的 MS-SSIM AUC 等权平均，再按第 4.3 节做 hierarchical bootstrap；单独 q 的排序只作次要分解。

### Gate G3

- VisDrone ordered prefix 的字节严格单调不减，且 MS-SSIM monotonic-violation rate 不超过全部 image-prefix transitions 的 1%；
- VisDrone ordered MS-SSIM AUC 相对每张图像其余 23 种顺序中位数的 hierarchical 95% CI 下界大于 0，并在三个 seed 中至少两个取得不低于第 6 名的 AUC rank；
- 在 VisDrone 四个 byte budgets 上分别计算 ordered 与每张图像 alternative median 的配对 MS-SSIM 差值并做 Holm 校正；至少 3/4 budgets 的 95% CI 下界须大于 0，且 ordered 聚合 AUC 相对 reverse-order 的 95% CI 下界须不小于 0；
- UAVid 的 ordered AUC 相对 alternative median 和 reverse-order 的聚合点估计均须不小于 0，作为跨场景方向一致性边界；
- 未通过时，删除“importance-ordered scalable latent”表述，只保留离散多档率控制。

### 输出

- `data/core_v3_progressive_ordering.csv`
- `data/core_v3_progressive_monotonicity.csv`
- `figures/fig_core_v3_progressive_trajectory.*`

## 9. E4：动态带宽与尾部等待时间

### 策略

- `BAA-controller`：风险感知的确定性 lookup controller。编码前，它依据 estimated effective goodput、packet-loss estimate、deadline、development size/latency calibration 和冻结的 10% goodput lower quantile 选择 `q_index`；编码一次后，再用实际 segment lengths 在同一安全预算内选择最高 development utility 的 prefix。安全预算目标是 development traces 上至少 90% 的 on-time probability。
- `fixed-r0.25/r0.50/r0.75/r1.00`：同一 checkpoint 的固定 prefix；每个 prefix 按全部 development profiles x deadlines 等权 mean `U_frame` 选择一个固定 `q_index`，之后不根据逐帧链路改变 action。
- `best-fixed`：从 16 个固定 `(active_ratio, q_index)` 中按全部 development profiles x deadlines 等权 mean `U_frame` 选择一个全局 action，测试时冻结；并列时使用下述统一 tie-break。
- `capacity-greedy`：使用与 BAA-controller 相同的 estimated goodput、deadline、development mean size/utility 和编码后的实际 segment lengths，但不使用 lower-quantile safety margin 或 packet-loss risk correction；选择 point-estimate payload 内 utility 最高的 action。
- `prefix-oracle`：逐帧事后上界，单独标注，不参与主要显著性结论；
- progressive OpenJPEG/JPEG2000：使用一个四层 LRCP stream；四个 cumulative layer targets 在 development tiles 上匹配 BAA-NIC `q_index=1` 的四个 ordered-prefix 中位实际 bpp，并使用 capacity-greedy layer selection。它只作为 operational reference，不进入 learned-codec RD 排名。

BAA-controller 与 capacity-greedy 必须共享同一 BAA action table 和信息时序：`q_index` 在编码前确定；每帧只允许一次 analysis transform 和每组一次 entropy encode；编码后只能依据实际 segment lengths 减少 transmitted prefix，禁止试编码其他 `q_index`。controller 在测试时不得访问当前/未来帧图像、image ID、latent、重建质量、profile ID、未来链路状态或测试统计，只能使用一帧延迟的链路估计、当前 deadline、调度器当时可见的因果 worker-availability 状态和冻结的 development calibration。fixed 策略不使用该编码后截断规则，但网络可自然交付 deadline 前完整到达的较低层；OpenJPEG 使用其独立的四层 action table，不参与 BAA action 选择。

对 action `a=(q,k)`，development utility 固定为 `u_dev(a)=mean_MS-SSIM_dev(a)`。将三个 development profiles 的 residual samples 合并，在 packet-loss estimate 的 `[0,0.05)/[0.05,0.15)/[0.15,1]` 三个 bins 内计算 `g_true/g_est` 的 10% quantile；bin 少于 30 个样本时退回 pooled global quantile，得到 profile-agnostic `g_0.10`。所有 goodput 使用 bit/s、时间使用 s、payload 使用 bytes。令候选 payload 长度为 `L(a)`，其线上字节为 `N_net(a)=L(a)+ceil(L(a)/1460)*40`。在 frame arrival `t_0`，调度器只依据当前已排队任务维护三个 worker 的因果可用时刻；BAA-controller 用 action-specific development calibration 计算 `t_enc_done=max(t_0,t_enc_available)+T_enc,p95(q)`、`t_tx_start=max(t_enc_done,t_tx_available)` 和 `t_last=deadline-T_dec,p95(q,k)`。仅当 `t_dec_available<=t_last` 时，安全线上预算为 `B_wire=max(0,g_est*g_0.10*(t_last-t_tx_start)/8)`，否则为 0。编码前用 development p90 `L(a)` 计算候选 `N_net(a)`，在 `N_net(a)<=B_wire` 的 action 中按冻结 utility 选 nominal `(q,k)`；编码后保持 q 不变，用实际累计 `L(q,k')` 选择不高于 nominal k 且满足预算的最高 utility prefix。capacity-greedy 使用同一调度状态和公式，但把风险修正及 p95/p90 calibration 分别替换为 point-estimated goodput、action-specific mean latency 和 mean payload。若编码前没有非空 action 可行，两个自适应策略均固定选择 `(q=3, nominal k=1)` 完成单次编码；编码后 base 仍不满足预算时置 `k=0`、不发送任何 bytes 且 `U_frame=0`。utility 差小于 `1e-6` 时依次选择 network bytes 更少、k 更小、q 更大的 action；仍相同时按 `(q,k)` 字典序选择，保证结果确定。

### Trace 设计

- 每个 realization 含 300 帧，帧率固定为 4 fps；deadline 为 100/250/500 ms。development 每个 profile 使用 200 个 realizations（seeds `110000-110199`），test 使用 1000 个 realizations（seeds `210000-210999`）；同一 realization 在三个 deadlines 和所有策略间配对复用。
- development realization 只使用 256 张 development tiles：由 realization seed 生成第一轮 256 张无放回排列，再由 `seed+10,000` 生成第二轮排列并取前 44 张，拼成 300 帧；test realization 才从 512 张 VisDrone test tiles 中无放回选择并排列 300 张。对应图像序列在全部策略、三个模型 seed 和三个 deadlines 间保持一致，development/test 图像绝不交叉。
- `smooth-AR`：log-goodput AR(1)，均值 4.0 Mbps、`phi=0.95`、innovation sigma 0.12，截断到 0.4-8.0 Mbps。
- `drop-recovery`：两状态 Markov 链，high/low 均值为 5.5/0.8 Mbps，`p(high->low)=0.03`、`p(low->high)=0.12`，状态内 log sigma 为 0.10/0.15，截断到 0.3-8.0 Mbps。
- `burst-loss`：raw capacity 为均值 4.5 Mbps、`phi=0.90`、innovation sigma 0.10 的 AR(1)；Gilbert-Elliott `p(good->bad)=0.02`、`p(bad->good)=0.20`，good/bad loss 分别从 `[0,0.02]`/`[0.12,0.30]` 均匀采样，effective goodput 只计算一次 `raw_capacity*(1-loss)`。
- controller 观测为一帧延迟的 effective goodput 乘以 log-normal estimation noise（log sigma 0.15）；frame 0 的 goodput estimate 固定为 pooled development effective-goodput median。packet-loss estimate 使用 `alpha=0.2` 的因果 EWMA并初始化为 0。测试生成器参数不得根据测试结果改变。
- replay 使用一个 encoder worker、一个 transmitter 和一个 decoder worker组成可并行流水线，三个 stage 各自 FCFS、单任务、非抢占；frame 到达 encoder queue 后依次进入 link 和 decoder queue，已过 deadline 的 frame 不再进入下一 stage。frame segments 按 prefix 顺序传输；MTU 1500 bytes，network payload 1460 bytes，每 packet 计 40 bytes 协议开销。effective goodput 已包含 loss 影响，不另行模拟 ARQ，避免双重扣减。
- deadline 从 frame arrival 开始，包含 encoder waiting/实际 encode latency、transmission waiting、逐 packet transmission、decoder waiting和实际 delivered-prefix decode latency。deadline 到达时丢弃该帧未完成 segment；base segment 未完整解码则记为 dropped，否则使用 deadline 前最高完整 prefix 的真实缓存重建。已开始的 encoder/decoder 非抢占任务完成后才释放 worker，但其过期输出丢弃。
- codec bitstreams 和重建离线缓存。除 E1 的 100-image 正式计时外，E4 还须在相同平台为 256 张 development tiles 与 512 张 VisDrone test tiles、每个模型 seed 和全部 q/prefix 生成一次同步后的 action-specific encode/decode latency cache；OpenJPEG 生成对应四层 cache。development cache 只提供 mean/p95 calibration，test cache 只在 replay 中按 image/action 注入实际 latency；二者分别保存 hash，replay 不重复神经推理且不得将 latency 置零。
- 所有策略使用逐条相同 traces、图像序列、packetization、队列和协议开销。该实验只支持“受控 stochastic trace replay”主张，不外推为真实 UAV 无线场测。

### 指标

- delivered-frame ratio、severe-link delivery、bitrate overshoot、deadline violation；
- encoder waiting、transmission waiting、decoder waiting 和 end-to-end latency 的 mean/p50/p95/p99/p99.9；
- unique-pixel throughput，重复 refinement bits 不重复计像素；
- delivered-frame PSNR/MS-SSIM 和主终点 `U_frame=delivered*MS-SSIM`；
- policy switch rate 和相邻帧 active-ratio variation。

`severe-link` 定义为各 profile development true effective-goodput 分布的最低 20%，阈值在 test 前冻结。`bitrate overshoot=max(0,selected_network_bytes-predicted_payload_bytes)/max(1,predicted_payload_bytes)`；`deadline violation` 为请求 action 未在 deadline 前完整交付的比例；`k=0` 帧的 selected bytes 与 overshoot 均为 0。transmission waiting 从 encode 完成到首 packet 开始服务计算，end-to-end latency 从 frame arrival 到 delivered-prefix decode 完成计算。

### Gate G4

- 250 ms 为预注册主 deadline。BAA-controller 的 `U_frame` 必须在至少 2/3 profiles 上同时显著优于 best-fixed，并在至少 1/3 profiles 上显著优于 capacity-greedy；检验使用 Holm 校正且 95% CI 下界大于 0。
- 对全部 3 profiles x 3 deadlines 等权聚合时，分别计算 BAA-controller 相对 best-fixed 和 capacity-greedy 的 paired `U_frame` 差值；两项 hierarchical-bootstrap 95% CI 下界均须不小于 0，不能只依赖聚合点估计通过。
- 相对 capacity-greedy，p99 waiting time 的非劣界为 `max(5 ms, 5%)`，overshoot 和 deadline-violation 的非劣界均为 1 percentage point；对应差值的 95% CI 上界必须低于界值。
- controller 选择必须直接来自代码路径，禁止用逐帧最优搜索替代；
- 未通过时，只能声称“能够响应带宽”，不能声称自适应策略优于固定策略。

### 输出

- `data/core_v3_bandwidth_summary.csv`
- `data/core_v3_bandwidth_detail.csv`
- `data/core_v3_bandwidth_statistics.csv`
- `figures/fig_core_v3_bandwidth_latency.*`

## 10. 冻结训练合同和最小矩阵

### 10.1. Encoder、entropy 与临时 decoder 联合训练

- 输入为 512 x 512 random crops，effective batch size 固定为 8；显存不足时只允许用 gradient accumulation 保持 effective batch，不改变优化目标。
- 前 2,000 steps 只使用 `(q_index=0, full-prefix)` 做 reconstruction warm-up；随后使用预生成、按 seed 固定的 balanced action schedule，使 16 个 `(q_index,prefix)` 在任意连续 1,600 steps 内各出现相同次数。
- 主损失固定为 `L = L1(x_hat,x) + MSE(x_hat,x) + 0.01*R_likelihood + 0.05*L_cross_rate`。`L_cross_rate` 比较相同 q 下当前 prefix 与 stop-gradient full-prefix；ROI、edge、teacher、detector 和 perceptual loss 权重均为 0。
- 主参数使用 AdamW，initial lr `1e-4`、weight decay `1e-4`；entropy quantiles 使用独立 Adam，lr `1e-3`。1,000-step linear warm-up 后 cosine decay 到 `1e-6`；AMP 开启。
- 最大 60,000 steps、最少 30,000 steps；每 1,000 steps 在固定 64-tile development monitor subset 和 balanced 16 actions 上快速验证，每 5,000 steps 在全部 256 development tiles 上完整验证。checkpoint 只按 full-development 同一损失的 mean 选择；连续 4 次 full validation 无改善则在最少步数后早停。
- 训练验证使用 differentiable likelihood bpp；每个候选 best checkpoint 必须通过 P0 practical-bitstream validation，最终论文只使用实际 bytes。

### 10.2. Heavy/light decoder-only 训练

- 对每个主 seed 冻结 analysis encoder、quantization schedule、group entropy codec 和 bitstream format，丢弃联合训练时的临时 decoder；heavy/light decoders 均从头初始化。
- 两个 decoder 使用同一预生成 minibatch、crop、q、prefix schedule和初始化 seed（`model_seed+1000`），相同 AdamW/lr/weight decay 和 AMP 设置；sender 输出在训练前缓存并逐字节复用。
- decoder loss 固定为 `L_dec=L1+MSE+0.05*L_cross_rate`，不含 rate term。最大 30,000 steps、最少 15,000 steps；每 500 steps 在固定 64-tile monitor subset 验证，每 2,500 steps 在全部 development tiles 验证。连续 4 次 full validation 无 mean MS-SSIM 改善则早停，并选择 full-development mean MS-SSIM 最高的 checkpoint。
- heavy/light 的更新次数允许因同一早停规则不同而不同，但均不得超过预注册最大值；同时报告达到 best checkpoint 的 steps 和 wall time。

### 10.3. 可复现性与成本门槛

- 每个 seed 保存代码 commit/hash、完整参数、环境、manifest hash、action schedule hash、checkpoint、训练/验证日志和停止原因。
- full 与 decoder-only 分支在远程提交前各运行一次 500-step benchmark，测量 throughput、显存和预计墙钟；该 benchmark 只用于 GPU/partition、gradient accumulation 和成本选择，不得改变损失、steps、早停或数据协议。
- 远程作业提交前记录 GPU 类型、卡数、预计时间和成本上限；失败重跑必须沿用同一 seed/config 并保留旧日志。

### 10.4. 最小训练和对照矩阵

| 分支 | 训练量 | 种子 | 服务的主张 |
|---|---:|---:|---|
| encoder + group entropy + 临时 heavy decoder | 最多 60k steps/run | 3 | C0、C1、C3 |
| heavy decoder（发送端冻结） | 最多 30k steps/run | 3 | C2 |
| light decoder（发送端冻结） | 最多 30k steps/run | 3 | C2 |
| dense-conv control | 无训练 | - | C1 |
| all 24 group orderings | 无训练 | 3 checkpoints | C3a |
| fixed/controller/capacity-greedy/OpenJPEG replay | 无训练 | 1000 test realizations/profile | C3b |

最小训练总量为 3 个 joint runs 和 6 个 decoder-only runs；其余均复用已缓存 bitstream。计划不训练独立 single-rate 强模型，也不声称渐进模型相对独立 single-rate 模型具有可忽略的 scalability penalty。

### 10.5. 产物 schema 与验收键

所有 CSV 必须包含 `contract_version`、`contract_sha256`、`code_hash` 和相应 `checkpoint_sha256/manifest_sha256`。模板在 G0a 前生成并由 `validate_result_csv.py` 严格校验；缺列、重复配对键、非有限主指标或 hash 不一致均为 failure。

| 产物 | 最小配对键 | 必需核心字段 |
|---|---|---|
| bitstream roundtrip | dataset, seed, image_id, q_index, subset_mask | prefix_k, bytes, bpp, decode_ok, crc_ok, byte_prefix_equal, reconstruction_sha256 |
| complexity/latency | model, component, side, device, image_id, repeat | threads, params, checkpoint_bytes, MACs/pixel, GFLOPs@512, peak_memory_bytes, latency_ms |
| decoder asymmetry | dataset, seed, image_id, q_index, prefix_k, decoder | bitstream_sha256, bytes, bpp, PSNR, MS-SSIM, LPIPS, decoder_params, decoder_MACs, latency_ms |
| progressive ordering | dataset, seed, image_id, q_index, ordering_id, budget_percent | received_group_mask, cumulative_bytes, bpp, PSNR, MS-SSIM, LPIPS, AUC, violation_flags |
| bandwidth detail | model_seed, profile, realization_id, deadline_ms, frame_id, policy | image_id, true/estimated_goodput, packet_loss, causal worker-availability times, wire_budget_bytes, q_index, nominal/requested/delivered_prefix, payload/network_bytes, stage waits/latencies, delivered, MS-SSIM, U_frame, calibration_sha256, trace_config_sha256 |
| bandwidth statistics | profile, deadline_ms, policy, comparator, metric | effect_size, CI_low, CI_high, raw_p, Holm_p, noninferiority_margin, gate_pass |

## 11. 执行顺序

1. 本地 `dl_env` 记录 Python/PyTorch/CUDA/CompressAI 环境，生成冻结 manifests/action schedules，并实现 practical codec、任意 subset mask 和 controller entrypoints。
2. 在本地 VisDrone/UAVid 各 2 张图上完成 `G0a structural smoke`、单 batch forward/backward 和 16-action coverage；未通过不得同步训练代码。
3. 在新 Slurm 主机检查 `sinfo/squeue`，分别完成 joint/decoder-only 500-step benchmark，记录兼容 GPU、预计墙钟和费用；benchmark 审查通过后才提交正式作业。
4. 三个 joint seeds 处于同一就绪阶段，可在成本/配额允许时独立并行；每个 selected checkpoint 先完成 `G0b evidence gate`。
5. 某一 seed 的 G0b 通过后，可并行启动该 seed 的 heavy/light decoder-only 分支以及 E1 bitstream/latency cache；不等待其他已独立就绪的 seed。
6. 三个 seed 的 heavy/light 全部通过各自完成审查后执行 E2；三个 G0b 和公共 cache 完成后，E3 与 E4 可并行执行。
7. 每个分支独立审查 checkpoint、日志、CSV schema、统计和 gate，记录 `pass/blocked/failed`。任一上游失败时不得启动依赖它的下游分支。
8. 远程训练每 30 分钟检查；完成后同步结果并执行账户级 Slurm、进程、同步和计费状态审计。
9. 只有 G0a、三个 G0b 和 G1-G4 全部完成后才更新论文摘要、贡献、公式、表图和 DOCX。

## 12. 主文最小呈现

- Table 1：BAA-NIC、dense control 和 CompressAI hyperprior 的完整发送端复杂度/延迟分解；ShiftLIC 仅列为带明确来源的文献上下文。
- Table 2：相同 bitstream 下 heavy/light decoder 的配对结果。
- Table 3：controller、capacity-greedy、best-fixed、fixed-low/high 和 OpenJPEG 的动态链路统计及 tail latency。
- Figure 1：轻量 encoder、分层 bitstream、重 decoder 的非对称系统。
- Figure 2：ordered/reverse 与 23 种 alternative-order progressive trajectories 及 rank distribution。
- Figure 3：bandwidth、payload、prefix、waiting time 和 frame delivery 时间序列。

主文不设置强 codec RD 排行榜。OpenJPEG 只用于说明标准渐进码流的 operational reference；已有强模型结果可留在补充材料，但不得驱动摘要或贡献结论。

## 13. 最终支持判定

- `G0a/G0b pass`：结构 smoke、最终实际码流、byte-prefix equality 和独立解码成立；
- `G1 pass`：完整发送端轻量化成立；
- `G2 pass`：同码流的服务器重解码补偿成立；
- `G3 pass`：ordered scalable latent 成立；
- `G4 pass`：动态带宽收益成立；
- 所有 gate 均绑定版本化 checkpoint、冻结 manifests、实际 bytes、3 seeds 和置信区间。

任一 gate 失败时只降级对应主张，不通过新增无关实验“补数量”。历史 `data/real_*.csv`、旧计划和失败分支全部保留，不覆盖。

## 14. 方法学参考文献

1. Alves de Oliveira et al. Reduced-Complexity End-to-End Variational Autoencoder for on Board Satellite Image Compression. Remote Sensing, 2021. https://doi.org/10.3390/rs13030447
2. He et al. ELIC: Efficient Learned Image Compression With Unevenly Grouped Space-Channel Contextual Adaptive Coding. CVPR, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.html
3. Bao et al. ShiftLIC: Lightweight Learned Image Compression With Spatial-Channel Shift Operations. IEEE TCSVT, 2025. https://doi.org/10.1109/TCSVT.2025.3556708
4. Lee et al. DPICT: Deep Progressive Image Compression Using Trit-Planes. CVPR, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.html
5. Hojjat et al. ProgDTD: Progressive Learned Image Compression With Double-Tail-Drop Training. CVPRW, 2023. https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Hojjat_ProgDTD_Progressive_Learned_Image_Compression_With_Double-Tail-Drop_Training_CVPRW_2023_paper.html
6. Kurka and Gunduz. Bandwidth-Agile Image Transmission With Deep Joint Source-Channel Coding. IEEE TWC, 2021. https://doi.org/10.1109/TWC.2021.3090048
7. Naseri et al. Deep learning-based image compression for wireless communications: impacts on robustness, throughput, and latency. npj Wireless Technology, 2026. https://doi.org/10.1038/s44459-025-00019-6
8. Liu et al. RS-LLIC: A Lightweight Learned Image Compression Model with Knowledge Distillation for Onboard Remote Sensing. IEEE TGRS, 2026. https://doi.org/10.1109/TGRS.2026.3668020
9. InterDigital Inc. CompressAI pretrained image-compression model zoo. https://interdigitalinc.github.io/CompressAI/zoo.html
10. OpenJPEG Project. OpenJPEG JPEG 2000 codec and API documentation. https://www.openjpeg.org/
