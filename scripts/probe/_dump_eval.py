#!/usr/bin/env python
import sys, json
d = json.load(open(sys.argv[1]))
on, ze = d["NM_on"], d["NM_zero"]
print(f"=== {sys.argv[1]} ===")
for k in ("first_token", "free_gen"):
    a = on["overall"][k]["acc"] * 100
    b = ze["overall"][k]["acc"] * 100
    n = on["overall"][k]["n"]
    print(f"  overall {k:11s}: {a:6.2f}% / {b:6.2f}% / {a-b:+6.2f}pp  (n={n})")
br_on, br_ze = on.get("by_recall", {}), ze.get("by_recall", {})
for kk in br_on:
    o, z = br_on[kk], br_ze[kk]
    print(f"  recall={kk:5s}: ft {o['first_token']['acc']*100:5.1f}/{z['first_token']['acc']*100:5.1f} (n={o['first_token']['n']:3d})"
          f"   fg {o['free_gen']['acc']*100:5.1f}/{z['free_gen']['acc']*100:5.1f} (n={o['free_gen']['n']:3d})")
