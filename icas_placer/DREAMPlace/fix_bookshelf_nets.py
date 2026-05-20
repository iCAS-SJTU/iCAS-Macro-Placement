from pathlib import Path

root = Path("benchmarks/IBM")

net_files = sorted(root.glob("ibm*/ibm*.nets"))

if not net_files:
    raise RuntimeError("No .nets files found under benchmarks/IBM/ibm*/")

for net_file in net_files:
    backup = net_file.with_suffix(net_file.suffix + ".bak")
    if not backup.exists():
        backup.write_text(net_file.read_text())

    new_lines = []
    changed = 0

    for line in net_file.read_text().splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()

        # skip header / statistic / net declaration / blank line
        if (
            not s
            or s.startswith("UCLA")
            or s.startswith("NumNets")
            or s.startswith("NumPins")
            or s.startswith("NetDegree")
            or s.startswith("#")
        ):
            new_lines.append(raw)
            continue

        # pin line without offset, e.g. "p198 B"
        if ":" not in raw:
            parts = s.split()
            if len(parts) >= 2:
                indent = raw[:len(raw) - len(raw.lstrip())]
                node_name = parts[0]
                pin_direct = parts[1]
                new_lines.append(f"{indent}{node_name} {pin_direct} : 0 0")
                changed += 1
            else:
                new_lines.append(raw)
        else:
            # pin line with ":" but no coordinates, e.g. "p198 B :"
            left, right = raw.split(":", 1)
            if len(right.strip().split()) < 2:
                new_lines.append(left.rstrip() + " : 0 0")
                changed += 1
            else:
                new_lines.append(raw)

    net_file.write_text("\n".join(new_lines) + "\n")
    print(f"[fixed] {net_file}: added offsets to {changed} pin lines")
