"""Eval JSON 摘要 — 一行打印 overall + read NM-on/zero/Δ。"""
import json
import sys
from pathlib import Path

paths = sys.argv[1:]
for p in paths:
    d = json.loads(Path(p).read_text())
    on = d["NM_on"]
    off = d["NM_zero"]
    name = Path(p).stem

    def fmt(a, b, n):
        return f"NM-on={a*100:5.1f}% NM-zero={b*100:5.1f}% Δ={(a-b)*100:+5.2f}pp n={n}"

    ot_a = on["overall"]["first_token"]["acc"]
    ot_b = off["overall"]["first_token"]["acc"]
    ot_n = on["overall"]["first_token"]["n"]
    og_a = on["overall"]["free_gen"]["acc"]
    og_b = off["overall"]["free_gen"]["acc"]
    og_n = on["overall"]["free_gen"]["n"]

    rt_a = on["by_recall"].get("read", {"first_token": {"acc": 0, "n": 0}})["first_token"]
    rt_a, rt_n = rt_a["acc"], rt_a["n"]
    rt_b = off["by_recall"].get("read", {"first_token": {"acc": 0}})["first_token"]["acc"]
    rg_a = on["by_recall"].get("read", {"free_gen": {"acc": 0, "n": 0}})["free_gen"]
    rg_a, rg_n = rg_a["acc"], rg_a["n"]
    rg_b = off["by_recall"].get("read", {"free_gen": {"acc": 0}})["free_gen"]["acc"]

    print(f"=== {name} ===")
    print(f"  overall  first-token  {fmt(ot_a, ot_b, ot_n)}")
    print(f"  overall  free-gen     {fmt(og_a, og_b, og_n)}")
    print(f"  *read*   first-token  {fmt(rt_a, rt_b, rt_n)}")
    print(f"  *read*   free-gen     {fmt(rg_a, rg_b, rg_n)}")
