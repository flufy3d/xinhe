"""
XinheConfig — 心核配置(单全局 QueryHead + HippoDelta 架构)

旧 v9.5 双 NeuralMemoryPair(per full-attn 层 Hippo+Neo)已删,目标架构:
  - 单全局 QueryHead 从 embedding/低层 h_pre 派生动态 q
  - 单全局 HippoDelta(Gated DeltaNet)持有 M(d_v, d_k)
  - MAC-R(W_mac)拼前缀 + MAL(W_mal)中后层残差
  - backbone addons:LoRA(qkvo) + per-layer K/V(可选)
"""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class XinheConfig:
    # --- Backbone ---
    backbone_type: str = "qwen"
    backbone_model_path: str = "./models/qwen3.5-0.8b"
    backbone_weights_path: str = ""
    hidden_size: int = 1024
    freeze_backbone: bool = True

    # --- 投影维度(d_total = n_heads * head_dim;单全局架构里 mem_dim 走 d_value 独立空间)---
    n_heads: int = 16               # 头数
    head_dim: int = 64              # d_head(d_total = n_heads * head_dim)

    # --- MAC: Memory As Context (paper Titans 形态)---
    # per-layer K/V:每个 full_attention 层独立的 K_pers/V_pers,直接拼接到该层 attention
    # 的 K/V cache(不经过 q/k/v 投影,paper Eq 11-13 形态)。
    # MAC-R 前缀长度统一由 n_query 控制,不再独立 n_mem_tokens。
    n_persistent_per_layer: int = 16  # 每个 full_attention 层独立的 K/V tokens 数

    # --- LoRA(frozen backbone 适配 MAC OOD 的根因修复)---
    # 与 MAC 是 producer/consumer 协同(MAC 放 prefix,LoRA 学怎么读),不抢梯度。
    # 0 禁用;>0 注入到 target_modules 指定的层
    lora_rank: int = 0
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 编译加速(只对 backbone 单 transformer 块,不包记忆模块。多卡 device_map="auto" 下不安全,自动跳过)。
    compile_backbone_layers: bool = False

    # --- 训练 ---
    # 单一术语:turn = 一个 user-asst pair(在 conversation.py 内编为 1 个 tensor)
    value_weight_cap: float = 1.0    # v9 默认 cap 到 1.0,等价取消 v8 的 VALUE 5x 加权
    turn_max_tokens: int = 256       # 单 turn token 上限
    max_turns_per_episode: int = 16  # 单 episode 最多几个 turn
    tbptt_turns: int = 4             # 每多少个 turn backward 一次
    batch_size: int = 4
    learning_rate: float = 3e-4
    plugin_lr_multiplier: float = 1.0   # memory 学习率 = learning_rate × multiplier
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    per_segment_checkpoint: bool = False  # v9 必须关:vmap(grad) ↔ checkpoint saved_hooks 冲突
    # NM-only aux loss 权重。mem_out 过 frozen lm_head 预测 value token,
    # 逼记忆通路学 key→value 映射(绕过 backbone retrieval)。0 = 关; ~0.5 = 跟主 loss 等权。
    nm_aux_weight: float = 0.0

    # --- Margin-Based Shortcut Suppression(直接 attack 'NM-on==NM-zero' 退化解)---
    # 每 turn 跑两次 forward:NM-on(主,有 grad)+ NM-zero(no_grad,baseline)。
    # 加 hinge:NM-on loss 必须比 NM-zero loss 至少低 margin,否则给 penalty。
    # 等价于 "backbone 关掉 mem 也能答对" → 推梯度让 mem 通路必须有用。
    # 不开 = 0,标准开 = 1.0;margin=0.5 表示要求 mem 提供 ≥0.5 nat 的 loss 下降。
    shortcut_suppression: bool = False
    shortcut_margin: float = 0.5
    shortcut_lambda: float = 1.0
    # 训练时随机关 mem(让 backbone 不能假设 mem 总在),0=关,0.5=半 step NM-zero forward。
    # 与 shortcut_suppression 互补:这个是 implicit curriculum,前者是 explicit penalty。
    memory_dropout: float = 0.0

    # === QueryHead 单全局记忆(目标架构;use_query_head 已固定 True,字段留作 yaml 兼容)===
    use_query_head: bool = True
    single_global_memory: bool = True   # 单 QueryHead + 单 HippoDelta(用户 2026-05-27 选定)
    d_key: int = 256                    # delta key 空间
    d_value: int = 128                  # delta value 空间(=mem_out 维)
    n_query: int = 16                   # QueryHead 输出 q 数 = MAC-R 前缀 token 数
    query_source_layer: int = 0         # 0=embedding(防 MAC-R 循环依赖);>0=backbone 前 j 层(消融);⚠ 当前 dead config
    query_pool: str = "mean"            # query 池化:"mean"(default)over 所有非 pad token / "last"(legacy)只用最后一个
    read_mode: str = "query_head"       # "query_head"(单全局 MAC/MAL)/ "per_layer_delta"(v19:每 full-attn 层 Delta read,回 delta-w-end 已证 95%+ 的机制)
    read_scale_init: float = -3.0       # per_layer_delta:每层 read 注入强度 σ(read_scale) 起步
    mal_inject_layer: int = -3          # MAL 残差注入层(该层输入 += α·W_mal(mem_out));-3/-4 消融
    mal_alpha_init: float = -3.0        # MAL α=σ(logit) 起步 σ(-3)≈0.05
    mem_type: str = "hippo"             # "hippo"(default,delta-rule)/ "cross_attn"(v16 softmax retrieval)
    mem_max_slots: int = 32             # cross_attn 仅:buffer 大小(circular,满后丢最旧)
    mem_write_pool: str = "mean"        # cross_attn 仅:write turn 池化方式 "mean"/"last"
    mac_disabled: bool = False          # v17:True 时禁 MAC-R 前缀(N_m=0,不 cat 到 input),验证 MAC=毒 假设
                                        # nm_aux 仍保留(W_mac 作纯监督信号,不进主路径);MAL 是 read 唯一通路
    global_write_layer: int = -1        # 单全局 write 挂第几个 full_attention 物理层(-1=最后一个)
    pause_mix_gate: bool = True         # 只走 HippoDelta + QueryHead + MAL,纯化变量
    lambda_div: float = 0.0             # q 多样性 contrastive aux 权重
    # Gated DeltaNet(plain delta / gated delta 切换)
    gated_delta: bool = False           # False=纯 delta(boxed,含删旧关联);True=加 g 遗忘门
    tau_k_init: float = 1.0             # delta key 温度 init(τ=1 → β<1 admissible)
    beta_bias_init: float = 0.0         # delta W_β bias init
    delta_backend: str = "auto"         # auto|fla|torch(训练自动强制 torch)
    spectral_norm_cap: float = 10.0     # delta M 谱范数监控上限(<10×初始)

    resume_from: str = ""
    early_stop_loss: float = 0.0
    early_stop_patience: int = 0
    early_stop_value: float = 0.995
    early_stop_tell: float = 0.0
    use_joint_early_stop: bool = False
    early_stop: dict = field(default_factory=dict)
    val_sets: list = field(default_factory=list)
    warmup_steps: int = 100
    max_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 10
    device: str = "cuda"
    dtype: str = "bfloat16"

    # --- 数据 ---
    train_path: str = "./data/train.jsonl"
    val_path: str = "./data/val.jsonl"

    # --- 日志 ---
    use_wandb: bool = False
    wandb_project: str = "xinhe"
    wandb_run_name: Optional[str] = None

    @classmethod
    def _load_and_merge(cls, path: str) -> dict:
        """递归加载 yaml,支持链式 base 继承"""
        from pathlib import Path

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if "base" in raw:
            base_path = Path(path).parent / raw.pop("base")
            base_raw = cls._load_and_merge(str(base_path))
            for section, values in raw.items():
                if isinstance(values, dict) and section in base_raw and isinstance(base_raw[section], dict):
                    base_raw[section].update(values)
                else:
                    base_raw[section] = values
            raw = base_raw

        return raw

    @classmethod
    def _resolve_curriculum(cls, raw: dict, config_path: str) -> list[dict]:
        """解析课程配置(同 v8)"""
        from pathlib import Path

        curriculum = raw.pop("curriculum", []) or []
        curriculum_file = raw.pop("curriculum_file", None)
        stage_overrides = raw.pop("stage_overrides", {})

        if curriculum_file:
            cur_path = Path(config_path).parent / curriculum_file
            with open(cur_path, "r", encoding="utf-8") as f:
                cur_raw = yaml.safe_load(f)
            training_defaults = cur_raw.get("training_defaults", {})
            curriculum = cur_raw.get("stages", [])
            for stage in curriculum:
                merged = dict(training_defaults)
                merged.update(stage.get("training", {}))
                stage["training"] = merged

        if stage_overrides:
            default_ov = stage_overrides.get("default", {})
            for stage in curriculum:
                name = stage["name"]
                specific_ov = stage_overrides.get(name, {})
                merged_ov = {**default_ov, **specific_ov}
                data_ov = {k[5:]: v for k, v in merged_ov.items() if k.startswith("data_")}
                training_ov = {k: v for k, v in merged_ov.items() if not k.startswith("data_")}
                if training_ov:
                    training = stage.setdefault("training", {})
                    training.update(training_ov)
                if data_ov:
                    data = stage.setdefault("data", {})
                    data.update(data_ov)

        return curriculum

    @classmethod
    def from_yaml(cls, path: str) -> tuple["XinheConfig", list[dict]]:
        raw = cls._load_and_merge(path)
        curriculum = cls._resolve_curriculum(raw, path)

        flat = {}
        mapping = {
            "backbone": {
                "type": "backbone_type",
                "model_path": "backbone_model_path",
                "weights_path": "backbone_weights_path",
                "hidden_size": "hidden_size",
                "freeze": "freeze_backbone",
            },
            "state": {
                "n_heads": "n_heads",
                "head_dim": "head_dim",
                "n_persistent_per_layer": "n_persistent_per_layer",
                "use_query_head": "use_query_head",
                "single_global_memory": "single_global_memory",
                "d_key": "d_key",
                "d_value": "d_value",
                "n_query": "n_query",
                "query_source_layer": "query_source_layer",
                "query_pool": "query_pool",
                "read_mode": "read_mode",
                "read_scale_init": "read_scale_init",
                "mal_inject_layer": "mal_inject_layer",
                "mal_alpha_init": "mal_alpha_init",
                "mem_type": "mem_type",
                "mem_max_slots": "mem_max_slots",
                "mem_write_pool": "mem_write_pool",
                "mac_disabled": "mac_disabled",
                "global_write_layer": "global_write_layer",
                "pause_mix_gate": "pause_mix_gate",
                "lambda_div": "lambda_div",
                "gated_delta": "gated_delta",
                "tau_k_init": "tau_k_init",
                "beta_bias_init": "beta_bias_init",
                "delta_backend": "delta_backend",
                "spectral_norm_cap": "spectral_norm_cap",
            },
            "training": {
                "value_weight_cap": "value_weight_cap",
                "turn_max_tokens": "turn_max_tokens",
                "max_turns_per_episode": "max_turns_per_episode",
                "tbptt_turns": "tbptt_turns",
                "batch_size": "batch_size",
                "learning_rate": "learning_rate",
                "plugin_lr_multiplier": "plugin_lr_multiplier",
                "weight_decay": "weight_decay",
                "grad_clip": "grad_clip",
                "grad_accum_steps": "grad_accum_steps",
                "gradient_checkpointing": "gradient_checkpointing",
                "per_segment_checkpoint": "per_segment_checkpoint",
                "nm_aux_weight": "nm_aux_weight",
                "shortcut_suppression": "shortcut_suppression",
                "shortcut_margin": "shortcut_margin",
                "shortcut_lambda": "shortcut_lambda",
                "memory_dropout": "memory_dropout",
                "compile_backbone_layers": "compile_backbone_layers",
                "resume_from": "resume_from",
                "early_stop_loss": "early_stop_loss",
                "early_stop_patience": "early_stop_patience",
                "early_stop_value": "early_stop_value",
                "early_stop_tell": "early_stop_tell",
                "use_joint_early_stop": "use_joint_early_stop",
                "early_stop": "early_stop",
                "warmup_steps": "warmup_steps",
                "max_steps": "max_steps",
                "eval_every": "eval_every",
                "save_every": "save_every",
                "log_every": "log_every",
                "device": "device",
                "dtype": "dtype",
            },
            "data": {
                "train_path": "train_path",
                "val_path": "val_path",
                "val_sets": "val_sets",
            },
            "logging": {
                "use_wandb": "use_wandb",
                "project": "wandb_project",
                "run_name": "wandb_run_name",
            },
            "lora": {
                "rank": "lora_rank",
                "alpha": "lora_alpha",
                "dropout": "lora_dropout",
                "target_modules": "lora_target_modules",
            },
        }

        for section, fields in mapping.items():
            if section in raw:
                for yaml_key, field_name in fields.items():
                    if yaml_key in raw[section]:
                        flat[field_name] = raw[section][yaml_key]

        return cls(**flat), curriculum
