"""
XinheConfig — 心核配置 (v5c)

包含状态机制、LoRA、训练等超参数。
v5c: Delta Rule 联想记忆 W: (B,H,d_v,d_k)，删除 n_state/state_dim/
     contrastive_weight/write_iterations；新增 n_heads/head_dim/
     beta_bias_init；state_scale_init → read_scale_init。
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

    # --- 持久状态 W_fact (v6: Delta Rule) ---
    n_heads: int = 16               # 头数
    head_dim: int = 64              # d_k = d_v = head_dim
    read_scale_init: float = -5.0   # sigmoid(-5) ≈ 0.007，空态几乎无影响
    beta_bias_init: float = 0.0     # sigmoid(0)=0.5，初始 Delta Rule 学习率

    # --- 持久状态 W_turn (v6: 自旋时序罗盘，与 W_fact 双流) ---
    enable_turn_memory: bool = True             # 默认开；yaml 可显式 false 做 ablation
    turn_read_scale_init: float = -3.0          # sigmoid(-3) ≈ 0.047（和 fact read_scale 同档次）。
                                                # -8 的静默启动在 Stage 0a 由 freeze_turn + W_turn=0 保证，
                                                # -3 让 0b/1 时 scale 的梯度（sigmoid'≈0.045）能被 LM loss 推动。
    turn_gamma: float = 0.9                     # 固定衰减（非 Parameter，config 超参）
    turn_rotation_base: float = 10000.0         # 旋转频率 base（同 Qwen RoPE）
    turn_phase_max: int = 5                     # 多相位共振搜索窗口 τ ∈ {0..phase_max}
    turn_phase_temperature: float = 5.0         # softmax 温度系数：score×K，K 越大选择越锐利（防止早期梯度稀释）
    turn_dtau_hidden: int = 128                 # [deprecated] 旧 Δτ 头隐层，多相位搜索后无用
    turn_lr_multiplier: float = 1.0             # turn 读侧参数 lr 倍率
    freeze_turn: bool = False                   # 0a_fact_bootstrap 用
    freeze_fact: bool = False                   # 0b_turn_bootstrap 用
    # 彻底屏蔽对应 interface 的 read_layer（跳过 forward 注入）。
    # freeze_* 只锁参数，但 read_layer 仍会执行并注入 sigmoid(scale)×(q·W) 噪声；
    # suppress_*_read 完全跳过，强制 clean 单流训练
    suppress_turn_read: bool = False            # 0a 用：完全不让 W_turn 影响 forward
    suppress_fact_read: bool = False            # 0b 用：逼模型只能用 W_turn 解题

    # --- LoRA ---
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- 训练 ---
    segment_length: int = 256
    episode_length: int = 16
    tbptt_steps: int = 4
    batch_size: int = 4
    learning_rate: float = 3e-4
    plugin_lr_multiplier: float = 1.0   # plugin 学习率 = learning_rate × multiplier
    freeze_lora: bool = False           # 冻结 LoRA，只训练 plugin (bootstrap 阶段用)
    lora_reset: bool = False            # persona 重训: 加载 plugin 但重新零初始化 LoRA
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1           # 梯度累积步数 (模拟更大 batch)
    gradient_checkpointing: bool = False  # 用计算换显存 (重算激活值)
    resume_from: str = ""               # checkpoint 路径 (为空则不恢复)
    # v5c: 早停基于 val VALUE，不再看 loss；下面两项保留字段避免破坏 YAML 兼容但实际无效
    early_stop_loss: float = 0.0        # (deprecated) 早停 loss 阈值 (0=不启用)
    early_stop_patience: int = 0        # (deprecated) 早停耐心
    early_stop_value: float = 0.995     # 早停 VALUE 阈值 (val breakdown 跨过即切下一 stage)
    early_stop_tell: float = 0.0        # 早停 TELL 阈值（整段 exact-match 率）；0 = 不查 TELL
    # persona 统一训练: 联合早停（legacy 4 + v6 新 4 = 8 指标）
    use_joint_early_stop: bool = False
    early_stop_world_qa: float = 0.70
    early_stop_refusal: float = 0.85
    early_stop_compositional: float = 0.85
    # v6 新 4 个阈值（0.0 = 跳过该指标的联合早停检查）
    early_stop_pronoun: float = 0.0
    early_stop_disentangle: float = 0.0
    early_stop_rapid_overwrite: float = 0.0
    early_stop_decay: float = 0.0
    val_worldqa_path: str = ""          # 世界 QA val jsonl（单轮 Q/A）
    val_refusal_path: str = ""          # Refusal val jsonl（多轮，每 ep 最后问未披露）
    val_compositional_path: str = ""    # Compositional val jsonl（多 fact 单 utterance）
    # v6 新 4 val 路径
    val_pronoun_path: str = ""          # 变距代词消解 val
    val_disentangle_path: str = ""      # fact vs transient val
    val_rapid_overwrite_path: str = ""  # 快速覆写 val
    val_decay_path: str = ""            # 遗忘感知 val
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
        """递归加载 yaml，支持链式 base 继承"""
        from pathlib import Path

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if "base" in raw:
            base_path = Path(path).parent / raw.pop("base")
            base_raw = cls._load_and_merge(str(base_path))
            # 深度合并: raw 覆盖 base_raw
            for section, values in raw.items():
                if isinstance(values, dict) and section in base_raw and isinstance(base_raw[section], dict):
                    base_raw[section].update(values)
                else:
                    base_raw[section] = values
            raw = base_raw

        return raw

    @classmethod
    def _resolve_curriculum(cls, raw: dict, config_path: str) -> list[dict]:
        """
        解析课程配置，支持两种方式:
          1. curriculum: [...] — 内联 (旧格式，向后兼容)
          2. curriculum_file: curriculum.yaml — 引用共享课程文件
             + training_defaults: 自动合并到每个阶段
             + stage_overrides: 硬件相关覆盖 (如 batch_size)
        """
        from pathlib import Path

        # 方式 1: 内联
        curriculum = raw.pop("curriculum", []) or []
        curriculum_file = raw.pop("curriculum_file", None)
        stage_overrides = raw.pop("stage_overrides", {})

        # 方式 2: 引用外部文件
        if curriculum_file:
            cur_path = Path(config_path).parent / curriculum_file
            with open(cur_path, "r", encoding="utf-8") as f:
                cur_raw = yaml.safe_load(f)
            training_defaults = cur_raw.get("training_defaults", {})
            curriculum = cur_raw.get("stages", [])

            # 合并 training_defaults → 每个阶段
            for stage in curriculum:
                merged = dict(training_defaults)
                merged.update(stage.get("training", {}))
                stage["training"] = merged

        # 合并 stage_overrides (硬件相关，优先级最高)
        # training 字段 → stage["training"], data 字段 → stage["data"]
        if stage_overrides:
            default_ov = stage_overrides.get("default", {})
            for stage in curriculum:
                name = stage["name"]
                specific_ov = stage_overrides.get(name, {})
                merged_ov = {**default_ov, **specific_ov}
                # data_ 前缀的 key 覆盖到 data 字段
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
        """
        从 yaml 配置文件加载，支持链式 base 继承。

        返回: (config, curriculum_stages)
            curriculum_stages: 课程学习阶段列表，无 curriculum 段时为空列表
        """
        raw = cls._load_and_merge(path)

        # 解析课程配置
        curriculum = cls._resolve_curriculum(raw, path)

        # 将嵌套的 yaml 扁平化到 dataclass 字段
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
                "read_scale_init": "read_scale_init",
                "beta_bias_init": "beta_bias_init",
                # --- v6 双流 W_turn ---
                "enable_turn_memory": "enable_turn_memory",
                "turn_read_scale_init": "turn_read_scale_init",
                "turn_gamma": "turn_gamma",
                "turn_rotation_base": "turn_rotation_base",
                "turn_phase_max": "turn_phase_max",
                "turn_phase_temperature": "turn_phase_temperature",
                "turn_dtau_hidden": "turn_dtau_hidden",     # [deprecated] 仅为兼容旧 yaml，不再使用
            },
            "lora": {
                "rank": "lora_rank",
                "alpha": "lora_alpha",
                "dropout": "lora_dropout",
                "target_modules": "lora_target_modules",
            },
            "training": {
                "segment_length": "segment_length",
                "episode_length": "episode_length",
                "tbptt_steps": "tbptt_steps",
                "batch_size": "batch_size",
                "learning_rate": "learning_rate",
                "plugin_lr_multiplier": "plugin_lr_multiplier",
                "turn_lr_multiplier": "turn_lr_multiplier",
                "freeze_lora": "freeze_lora",
                "freeze_turn": "freeze_turn",
                "freeze_fact": "freeze_fact",
                "suppress_turn_read": "suppress_turn_read",
                "suppress_fact_read": "suppress_fact_read",
                "lora_reset": "lora_reset",
                "weight_decay": "weight_decay",
                "grad_clip": "grad_clip",
                "grad_accum_steps": "grad_accum_steps",
                "gradient_checkpointing": "gradient_checkpointing",
                "resume_from": "resume_from",
                "early_stop_loss": "early_stop_loss",
                "early_stop_patience": "early_stop_patience",
                "early_stop_value": "early_stop_value",
                "early_stop_tell": "early_stop_tell",
                "use_joint_early_stop": "use_joint_early_stop",
                "early_stop_world_qa": "early_stop_world_qa",
                "early_stop_refusal": "early_stop_refusal",
                "early_stop_compositional": "early_stop_compositional",
                "early_stop_pronoun": "early_stop_pronoun",
                "early_stop_disentangle": "early_stop_disentangle",
                "early_stop_rapid_overwrite": "early_stop_rapid_overwrite",
                "early_stop_decay": "early_stop_decay",
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
                "val_worldqa_path": "val_worldqa_path",
                "val_refusal_path": "val_refusal_path",
                "val_compositional_path": "val_compositional_path",
                # v6 新 val 路径
                "val_pronoun_path": "val_pronoun_path",
                "val_disentangle_path": "val_disentangle_path",
                "val_rapid_overwrite_path": "val_rapid_overwrite_path",
                "val_decay_path": "val_decay_path",
            },
            "logging": {
                "use_wandb": "use_wandb",
                "project": "wandb_project",
                "run_name": "wandb_run_name",
            },
        }

        for section, fields in mapping.items():
            if section in raw:
                for yaml_key, field_name in fields.items():
                    if yaml_key in raw[section]:
                        flat[field_name] = raw[section][yaml_key]

        return cls(**flat), curriculum
