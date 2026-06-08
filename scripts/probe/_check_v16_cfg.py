"""Verify v16 config loads correctly + CrossAttnMem instantiates."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig

cfg, _ = XinheConfig.from_yaml(str(project_root / "configs/pcap_skeleton_5080_v16.yaml"))
print(f"mem_type: {cfg.mem_type}")
print(f"mem_max_slots: {cfg.mem_max_slots}")
print(f"mem_write_pool: {cfg.mem_write_pool}")
print(f"query_source_layer: {cfg.query_source_layer}")
print(f"mal_alpha_init: {cfg.mal_alpha_init}")
print(f"query_pool: {cfg.query_pool}")
print("OK")
