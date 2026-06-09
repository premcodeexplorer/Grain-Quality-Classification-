import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

BASE = r"E:/GrainQuality/tiny_data"
GRAINS = ["maize", "rice", "sorg", "wheat"]
LABELS = ["NOR", "BN", "MY", "SD", "AP", "HD", "UN", "IM", "F&S", "BP"]

def parse_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for obj in root.findall("object"):
        rows.append({
            "ID": obj.findtext("ID", "").strip(),
            "species": obj.findtext("species", "").strip(),
            "DU_grain": obj.findtext("DU_grain", "").strip(),
        })
    return rows

def collect_folder_ids(grain):
    ids = set()
    for split in ("train", "test"):
        root = os.path.join(BASE, grain, split)
        if not os.path.isdir(root):
            continue
        for cls in os.listdir(root):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                stem, _ = os.path.splitext(f)
                ids.add(stem)
    return ids

xml_data = {}
for g in GRAINS:
    xml_data[g] = parse_xml(os.path.join(BASE, f"{g}_tiny.xml"))

# 1) totals per grain (from XML)
print("=" * 70)
print("1) TOTAL IMAGES PER GRAIN (from XML)")
print("=" * 70)
xml_totals = {}
for g in GRAINS:
    xml_totals[g] = len(xml_data[g])
    print(f"  {g:6s}: {xml_totals[g]:6d} entries")
print(f"  {'TOTAL':6s}: {sum(xml_totals.values()):6d}")

# 2) per-grain DU_grain label counts
print("\n" + "=" * 70)
print("2) DU_grain LABEL COUNTS PER GRAIN")
print("=" * 70)
label_counts = {}
for g in GRAINS:
    c = Counter(r["DU_grain"] for r in xml_data[g])
    label_counts[g] = c

all_labels = sorted({lbl for c in label_counts.values() for lbl in c})
header = f"  {'Label':6s} | " + " | ".join(f"{g:>6s}" for g in GRAINS) + " | " + f"{'TOTAL':>6s}"
print(header)
print("  " + "-" * (len(header) - 2))
totals_per_label = Counter()
for lbl in all_labels:
    row_vals = [label_counts[g].get(lbl, 0) for g in GRAINS]
    total = sum(row_vals)
    totals_per_label[lbl] = total
    print(f"  {lbl:6s} | " + " | ".join(f"{v:>6d}" for v in row_vals) + f" | {total:>6d}")

# 3) check folder vs XML IDs
print("\n" + "=" * 70)
print("3) FOLDER vs XML ID MATCH")
print("=" * 70)
for g in GRAINS:
    folder_ids = collect_folder_ids(g)
    xml_ids = {r["ID"] for r in xml_data[g]}
    inter = folder_ids & xml_ids
    only_folder = folder_ids - xml_ids
    only_xml = xml_ids - folder_ids
    print(f"  {g}: folder={len(folder_ids)}, xml={len(xml_ids)}, "
          f"match={len(inter)}, only_folder={len(only_folder)}, only_xml={len(only_xml)}")

# 4) class distribution percentages (overall across grains)
print("\n" + "=" * 70)
print("4) CLASS DISTRIBUTION (%) — per grain")
print("=" * 70)
header = f"  {'Label':6s} | " + " | ".join(f"{g:>8s}" for g in GRAINS)
print(header)
print("  " + "-" * (len(header) - 2))
for lbl in all_labels:
    parts = []
    for g in GRAINS:
        total = xml_totals[g]
        pct = 100.0 * label_counts[g].get(lbl, 0) / total if total else 0.0
        parts.append(f"{pct:>7.2f}%")
    print(f"  {lbl:6s} | " + " | ".join(parts))

# 5) imbalance analysis
print("\n" + "=" * 70)
print("5) CLASS IMBALANCE ANALYSIS")
print("=" * 70)
for g in GRAINS:
    c = label_counts[g]
    total = sum(c.values())
    mx_lbl, mx = max(c.items(), key=lambda kv: kv[1])
    mn_lbl, mn = min(c.items(), key=lambda kv: kv[1])
    ratio = mx / mn if mn else float("inf")
    rare = [(lbl, v) for lbl, v in c.items() if v < 0.05 * total]
    print(f"\n  {g.upper()} (n={total}):")
    print(f"    majority: {mx_lbl} ({mx}, {100*mx/total:.1f}%)")
    print(f"    minority: {mn_lbl} ({mn}, {100*mn/total:.1f}%)")
    print(f"    imbalance ratio (max/min): {ratio:.1f}x")
    if rare:
        print(f"    under-represented (<5%): " +
              ", ".join(f"{l}={v}" for l, v in rare))
