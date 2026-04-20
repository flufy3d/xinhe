"""
XinheConfig — 心核配置

包含状态机制、LoRA、训练等全部超参数。
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

    # --- 持久状态 ---
    n_state: int = 32               # 状态 token 数
    state_dim: int = 1024           # 状态维度 (可独立于 hidden_size)
    state_scale_init: float = -5.0  # sigmoid(-5) ≈ 0.007，空状态几乎无影响

    # --- EKS (Entity-Keyed State, v4) ---
    temperature_init: float = 1.0   # routing softmax 温度初值 (learnable, clamp min=0.3)
    eks_alpha_init: float = -5.0    # 新旧路径混合: sigmoid(-5)≈0 → 开局完全走 v2 路径 (续训友好)
    entropy_aux_weight: float = 0.01  # routing entropy 正则权重 (0 关闭, >0 防 slot collapse)

    # --- Sleep (对话 buffer replay + 权重更新，里程碑 9 实现) ---
    sleep_every: int = 4            # 每隔几轮对话触发 sleep

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
    plugin_lr_multiplier: float = 1.0  # plugin 学习率 = learning_rate × multiplier
    plugin_core_lr_multiplier: float = 1.0  # plugin core 学习率额外乘数 (迁移时设 0.1)
    slot_attn_lr_multiplier: float = 3.0  # slot_attn 学习率 = learning_rate × plugin_mult × core_mult × slot_attn_mult (续训从恒等激活需要更高 LR)
    freeze_lora: bool = False           # 冻结 LoRA，只训练 plugin
    freeze_plugin_core: bool = False    # 冻结 plugin 核心参数 (迁移时只训投影层)
    train_only_slot_attn: bool = False  # 只训 slot_attn (冻结 LoRA + 所有 plugin 其他参数); D2 诊断用
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 1        # 梯度累积步数 (模拟更大 batch)
    gradient_checkpointing: bool = False  # 用计算换显存 (重算激活值)
    resume_from: str = ""            # checkpoint 路径 (为空则不恢复)
    early_stop_loss: float = 0.0    # 早停 loss 阈值 (0=不启用)
    early_stop_patience: int = 0    # 早停耐心 (连续多少步低于阈值)
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
            data_defaults = {}
            if "think_lang" in cur_raw:
                data_defaults["think_lang"] = cur_raw["think_lang"]
            curriculum = cur_raw.get("stages", [])

            # 合并 training_defaults / data_defaults → 每个阶段
            for stage in curriculum:
                merged = dict(training_defaults)
                merged.update(stage.get("training", {}))
                stage["training"] = merged
                if data_defaults:
                    data = stage.setdefault("data", {})
                    for k, v in data_defaults.items():
                        data.setdefault(k, v)

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
                "n_state": "n_state",
                "state_dim": "state_dim",
                "state_scale_init": "state_scale_init",
                "temperature_init": "temperature_init",
                "eks_alpha_init": "eks_alpha_init",
                "entropy_aux_weight": "entropy_aux_weight",
            },
            "sleep": {
                "sleep_every": "sleep_every",
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
                "plugin_core_lr_multiplier": "plugin_core_lr_multiplier",
                "slot_attn_lr_multiplier": "slot_attn_lr_multiplier",
                "freeze_lora": "freeze_lora",
                "freeze_plugin_core": "freeze_plugin_core",
                "train_only_slot_attn": "train_only_slot_attn",
                "weight_decay": "weight_decay",
                "grad_clip": "grad_clip",
                "grad_accum_steps": "grad_accum_steps",
                "gradient_checkpointing": "gradient_checkpointing",
                "resume_from": "resume_from",
                "early_stop_loss": "early_stop_loss",
                "early_stop_patience": "early_stop_patience",
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
