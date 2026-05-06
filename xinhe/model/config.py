"""
XinheConfig — 心核配置 (v9)

v9: 双 NeuralMemory(Hippocampus 浅 MLP + Neocortex 深 MLP),挂在每个 full-attn 层。
v8 的 read_scale / beta_proj / Delta-Rule W 已废弃;LoRA 已抛弃,backbone 全冻。
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

    # --- v9 NeuralMemoryPair ---
    n_heads: int = 16               # 头数
    head_dim: int = 64              # d_head(d_total = n_heads * head_dim)
    hippo_mlp_depth: int = 2
    hippo_mlp_expansion: float = 2.0
    neo_mlp_depth: int = 4
    neo_mlp_expansion: float = 4.0
    hippo_retention: float = 0.99    # Hippo 每 chunk 保留 99%(自然遗忘)
    hippo_base_lr: float = 1e-2      # Hippo 内层 test-time SGD lr 上限
    # NOTE: Neo 走标准 backprop,无 retention 概念;lr = outer learning_rate × plugin_lr_multiplier。
    mem_chunk_size: int = 64
    alpha_logit_init: float = -5.0   # sigmoid(-5)≈0.007 保守起步
    alpha_min_clamp: float = 0.02    # 防 alpha collapse 到 0
    gate_entropy_lambda: float = 0.01  # gate 熵正则,防单边塌缩
    phase: str = "P-cap"             # "P-cap" | "Operational",决定 Neo 默认 daytime_plastic

    # v9 freeze flags
    freeze_alpha: bool = False       # 测试基线时可冻 alpha
    freeze_gate_q: bool = False      # 测试基线时可冻 gate

    # 编译加速(只对 backbone 单 transformer 块,不包 NeuralMemoryPair —
    # Hippo 的 vmap+grad inner SGD 跟 Dynamo 的 saved_tensors_hooks 冲突,
    # 把 NM 排除在 compile 边界外即可。多卡 device_map="auto" 下不安全,自动跳过)。
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
                "hippo_mlp_depth": "hippo_mlp_depth",
                "hippo_mlp_expansion": "hippo_mlp_expansion",
                "neo_mlp_depth": "neo_mlp_depth",
                "neo_mlp_expansion": "neo_mlp_expansion",
                "hippo_retention": "hippo_retention",
                "hippo_base_lr": "hippo_base_lr",
                "mem_chunk_size": "mem_chunk_size",
                "alpha_logit_init": "alpha_logit_init",
                "alpha_min_clamp": "alpha_min_clamp",
                "gate_entropy_lambda": "gate_entropy_lambda",
                "phase": "phase",
            },
            "training": {
                "value_weight_cap": "value_weight_cap",
                "turn_max_tokens": "turn_max_tokens",
                "max_turns_per_episode": "max_turns_per_episode",
                "tbptt_turns": "tbptt_turns",
                "batch_size": "batch_size",
                "learning_rate": "learning_rate",
                "plugin_lr_multiplier": "plugin_lr_multiplier",
                "freeze_alpha": "freeze_alpha",
                "freeze_gate_q": "freeze_gate_q",
                "weight_decay": "weight_decay",
                "grad_clip": "grad_clip",
                "grad_accum_steps": "grad_accum_steps",
                "gradient_checkpointing": "gradient_checkpointing",
                "per_segment_checkpoint": "per_segment_checkpoint",
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
        }

        for section, fields in mapping.items():
            if section in raw:
                for yaml_key, field_name in fields.items():
                    if yaml_key in raw[section]:
                        flat[field_name] = raw[section][yaml_key]

        return cls(**flat), curriculum
