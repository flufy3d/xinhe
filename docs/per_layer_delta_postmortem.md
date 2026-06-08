# per-layer Delta 记忆线复盘(v9.5 → iter4-14 → v10-v22)

> 本文件随 **`archive/per-layer-delta-v22`** 备份分支归档。
> 它记录了 `delta-w-end` 之后那 30 个提交(整条 per-layer Delta / MAC+MAL 研究线)的全部进展、结论与失败教训。
> 写于 2026-06-08。main 在此之后回退到 `delta-w-end` 作为干净起点。

---

## TL;DR 终判

- **机制是通的**:per_layer_delta 记忆在真 checkpoint 上能召回,且召回 **100% 由记忆驱动**(NM-zero 消融下 read 恒为 0%,backbone 不偷记)。这本身破解了之前八版的 `read=0%` 诅咒。
- **但真实天花板只有 ~16%**:正确口径(`validate_memory` by_recall=read)下,历史最佳 v20 step14000 = **read first-token 16.4% / free-gen 14.5%**,离 95% 目标很远。
- **驱动整条线的"scaling law"是假的**:v19→v22 一整轮"纯 scale 数据/步数冲 95"的方向,根因是一个 **eval 度量 bug**(探针数到了模板 preamble token,不是 value token),把召回虚高到 ~63%。真口径一直只有 ~16%。
- **纯 visit-scaling 已证伪**:v21 把训练拉到 40000 步,read 不升反降(16.4→9.1 ft),是真过拟合。
- **shortcut suppression 可以去掉且更好**:v22(shortcut OFF)step6000 = 14.5% ft,追平 v20(ON)step14000,且同步 step6000 时 +5.4pp。per_layer_delta 不需要 shortcut 也不会 bypass。
- **决策**:这条线在 0.8B backbone + 当前 read 架构上接近联合天花板;回退 main 到 `delta-w-end`(用户确认曾达 95-98% recall 的真召回机制),从那条"能召回但换问法会崩"的线重新出发。

---

## 1. 架构回顾(per_layer_delta)

- backbone:Qwen3.5-0.8B(787M,hybrid full/linear attention,24 层,full-attn hook 层 `[3,7,11,15,19,23]`)。
- 记忆:单个 Hippocampus 权重 `W (B,16,128,128)`。
  - **READ**:6 个 full-attn hook 层各自 per-layer q/o projection,把 `W·q` 注入 hidden(`sigmoid(read_scale)` 门控,init=0)。
  - **WRITE**:每个 episode segment 末尾一次 Delta Rule 更新 W。
- 关掉的部件:Neo MLP / MAC 前缀 / MAL / 单 QueryHead read(`n_persistent_per_layer=0`)。
- 可训练参数 30.46M(LoRA r16/α32 + Hippocampus 投影)。
- NM-zero 消融:`mem_alpha_override=0.0` → read 注入全关,得纯 backbone baseline。

## 2. 三口径 strict eval(`scripts/validate_memory.py` —— 唯一可信基准)

- **first-token**:teacher-forced,prefix forward 后 `argmax==value 首 token`(用 char_span + offset_mapping 精确定位 value 串)。
- **free-gen**:greedy 解码整句,子串匹配 value。
- **NM-zero**:同输入关掉记忆注入,read 必须掉到 0 才证明召回来自记忆。
- **by_recall**:把 turn 分成 write(value 在 prompt 里,trivial)与 read(真召回)。**只有 read 口径可信。**

## 3. 实验时间线与关键数字

> 全部为 `validate_memory` by_recall=read,val 未见实体,NM-zero 恒 0%。

| 版本 | 配置要点 | read ft | read fg | 结论 |
|---|---|---|---|---|
| iter4 (MAC+MAL) | `mem_out_real_alpha=0.1` | recall_probe 10→100% | — | 机制在玩具集能 memorize,但平移到 skeleton 全 0 |
| iter4-14 | chunk/tbptt/alpha/cap/Neo/Q-K 单变量 | read=0% | 0% | 六轮单变量全失败;证伪 tbptt 距离、Q-K 裸用 |
| v10 step1000 | LoRA r4 + shortcut + 1000ep×3v | 0% | 0% | 完全 memorize 崩塌,转 v11(num_train 10000) |
| v11 step3000 | shortcut + 10k 数据 + NaN-skip | 0% | 0% | anti-collapse 生效但 read 仍是 capability ceiling |
| v13 step1000 | mal_alpha_init -3→0 | 1.8% | 1.8% | read 首破 0,验证 NM 残差强度是瓶颈 |
| v14 step3000 | mean-pool query | 0% (ft) | — | overall Δ+4.1pp 史上最佳,但 read ft 仍 0 |
| v15/v16 | backbone-aware query / cross-attn softmax | 0% | 0% | 架构改造仍 0,确认联合 ceiling |
| v17 | MAC disabled | — | — | 反证 MAC 是毒;锁定根因=novel value 不泛化到 decode |
| v19/v19b | per_layer_delta(嫁接 delta-w-end read) | 0% | 0% | 机制健康(100 步 memorize 100%),探针误报 scaling |
| **v20 step14000** | per_layer_delta + shortcut + num_train2000 | **16.4%** | **14.5%** | **历史最佳真召回** |
| v21 step18000 | v20 续训到 40000 步 | 16.4% | 7.3% | fg 开始掉 |
| v21 step32000 | 续训(低 LR anneal) | 9.1% | 3.6% | **visit-scaling 过拟合证伪** |
| v22 step6000 | = v20 但 **shortcut OFF** | 14.5% | 5.5% | 6000 步追平 v20 14000 步 ft;shortcut 可去 |

## 4. 最重要的发现 —— "63% scaling law" 是度量 bug

驱动 v19→v22 整轮"纯 scale 冲 95"的那条漂亮曲线(探针 val read 21/29.5/50/63% @N=30/150/500/1000)**是假的**:

- `scripts/probe/nm_generalize.py` 的 `eval_set` 用 `vmask = weights > 0.5` 找 value 首 token;
- 但 `tokenize_turn`(`xinhe/data/conversation.py` ~line198)给**所有** assistant token 基础 `lm_weight=1.0`,value 只是再加权 `weight_per_span`(2.5/5.0);
- 于是 `vmask.argmax` 落在**第一个 assistant token = 模板 preamble**(如"常用"/"不是"/"您的"),而非 value;
- 预测 preamble 极易 → 虚高到 ~63-67%。

**逐 token 检视铁证如山**(`_inspect_localize.py`):eval_set 数的是开头模板词,`validate_memory._check_first_token`(char_span 精确定位)数的才是 value 串本身 → ~16%。修正口径(`weights > 1.5`)后探针掉回 ~11%,与 validate_memory 一致。

> 教训(已写入 memory `feedback_eval_metric_must_target_value_token`):任何 recall 口径上线前,必须逐 read-turn 打印「gold token 解码 + pred 解码」抽检它指的是不是 value 串本身。NM-zero≈0 只证"非 backbone 偷记",**不**证"度量的是 value"。validate_memory 可信,nm_generalize.eval_set 的 first-token 不可信。

## 5. 已证实 / 已证伪

**证实(真,正确口径量过):**
1. per_layer_delta 记忆机制通,召回 100% 由记忆驱动(NM-zero=0)。
2. 真实召回天花板 ~16% ft / ~14.5% fg(v20 step14000)。
3. shortcut suppression 在 per_layer_delta 上没必要,去掉反而更快到顶(v22)。

**证伪:**
1. "纯 visit-scaling 冲 95" —— v21 过拟合,read 随 visit 下降。
2. "probe 63% scaling law → 堆数据冲 95" —— 度量 bug,从无有效证据。
3. "shortcut 是 per_layer_delta 必需" —— 去掉不 bypass。

**未验证(被中止的 sweep 本要回答):** 正确口径下召回随 entity 多样性 N 是否真上升。这是通往 95% 的关键未知;若也平在 ~16%,说明 0.8B + 当前 read 架构是联合天花板,必须换能力杠杆(更大 backbone / 更强 read 架构 / value-token identity 编码)。

## 6. 为何回退 `delta-w-end`

- per_layer_delta 这条线把"read=0% 诅咒"破到了 16%,但再难往上,且证明了之前以为的几条路(visit-scaling、entity-scaling 的假证据、shortcut 必需)都站不住。
- `delta-w-end`(`04a2418`)是用户确认曾达 **95-98% recall** 的 per-layer 全参 Delta read 真机制,唯一毛病是"换问法就崩"(`docs/failure_postmortem.md` 定性为数据问题:v5c 课程 100% 窄合成模板 + 无 shortcut → LoRA 过拟合成模板填值)。
- 决策:回到那条"召回能起来"的线,用本 repo 已建好的泛化武器(shortcut suppression、多样化数据、三口径 eval、delta 数值约束)治它的"换问法崩",而不是继续在 16% 的 per_layer_delta 上磨。

## 7. 资产清单(本备份分支)

- **代码**:`xinhe/model/xinhe_model.py`(per_layer_delta forward 路径)、`config.py`(read_mode/read_scale_init 等字段)、`trainer.py`(shortcut + NaN-skip)、`conversation.py`、`evaluate.py` 等 12 个改动文件。
- **配置**:`configs/pcap_skeleton_5080_v2~v22.yaml` + `curriculum_pcap_skeleton_gen_*.yaml` 全套。
- **探针/脚本**:`scripts/probe/` 下全部诊断脚本(nm_generalize、_probe_*、_inspect_localize、validate 配套等)—— 编码了这一年的诊断方法学。
- **训练好的 checkpoint**(留在本地磁盘 `checkpoints/`,gitignored,**未删**):`xinhe_step_14000.pt`(v20,16.4/14.5)、`xinhe_step_6000.pt`(v22,shortcut-off 配方)。eval JSON 结果也在 `checkpoints/`(未入 git)。

## 8. 下一步候选方向(回到 delta-w-end 之后)

1. **delta-w-end 真机制 + 当前泛化武器**:把 95-98% recall 的 per-layer 全参 Delta read 配上 shortcut suppression + 多样化数据 + 改写 query,正面治"换问法崩"。这是 v19 计划的初衷,但 v19 把 read 压缩成了单点注入退化版;回到全参版本重做。
2. **能力杠杆**(若召回仍卡):更大 backbone(4B)、更强 read 架构、value-token identity 显式编码。
3. **先验证再 scale**:严守 gate 纪律(≤1000 步 skeleton-only,read free-gen > 0 且 decode rank 暴跌才放算力),并对任何新口径先做 value-token 定位抽检。
