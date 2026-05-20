#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert cluster-level .name/.net files to UCLA/UCSC placer input files.

Input format expected:
    <benchmark>.name
        Width: <outline_width>
        Height: <outline_height>
        Cluster_Name: <cluster_name>
        Cluster_num: <number_of_instances_in_cluster>
        Cluster_area: <area>
        ... repeated ...

    <benchmark>.net
        Net_1:
        source: <src> <dst1> <weight1> <dst2> <weight2> ...
        Net_2:
        source: ...
        ... repeated ...

Output files:
    <benchmark>.blocks.txt
    <benchmark>.nets.txt
    <benchmark>.pl
    <benchmark>.cluster_map.csv

Main assumptions:
    1. Each cluster is treated as the minimum placement unit.
    2. Each cluster is converted to a square hardrectilinear block.
    3. Width/Height in the .name file are treated as the placement outline.
    4. All block sizes and coordinates written to output are integers.
    5. Four dummy terminals are generated at the four outline corners:
           p1: left-bottom  (0, 0)
           p2: right-bottom (W, 0)
           p3: left-top     (0, H)
           p4: right-top    (W, H)
    6. The first net in the generated .nets.txt is a terminal net connecting p1/p2/p3/p4.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

@dataclass
class Cluster:
    original_name: str
    placer_name: str
    cluster_num: int
    area: float
    side: int


@dataclass
class NameFileData:
    clusters: List[Cluster]
    raw_width: Optional[float]
    raw_height: Optional[float]
    outline_width: Optional[int]
    outline_height: Optional[int]


@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    weight: float


TERMINALS = [
    ("p1", "left-bottom"),
    ("p2", "right-bottom"),
    ("p3", "left-top"),
    ("p4", "right-top"),
]


_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_WIDTH_RE = re.compile(r"\bWidth\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_HEIGHT_RE = re.compile(r"\bHeight\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def is_number(token: str) -> bool:
    return bool(_NUM_RE.match(token))


def to_int_coord(value: float, coord_scale: float) -> int:
    """
    Convert coordinate value to integer output coordinate.

    Example:
        Width: 22.95, coord_scale=100 -> 2295
        Height: 23.04, coord_scale=100 -> 2304
    """
    return int(round(float(value) * float(coord_scale)))


def sanitize_name(name: str) -> str:
    """Make a name safer for simple UCLA-style parsers."""
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.$@:/+-]", "_", name)
    if not name:
        name = "unnamed"
    if name[0].isdigit():
        name = "n_" + name
    return name


def parse_outline(lines: List[str], coord_scale: float) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    raw_width: Optional[float] = None
    raw_height: Optional[float] = None

    for line in lines:
        width_match = _WIDTH_RE.search(line)
        if width_match is not None:
            raw_width = float(width_match.group(1))

        height_match = _HEIGHT_RE.search(line)
        if height_match is not None:
            raw_height = float(height_match.group(1))

    outline_width = to_int_coord(raw_width, coord_scale) if raw_width is not None else None
    outline_height = to_int_coord(raw_height, coord_scale) if raw_height is not None else None

    if outline_width is not None and outline_width <= 0:
        raise ValueError(f"Parsed non-positive Width from .name: {raw_width} -> {outline_width}")
    if outline_height is not None and outline_height <= 0:
        raise ValueError(f"Parsed non-positive Height from .name: {raw_height} -> {outline_height}")

    return raw_width, raw_height, outline_width, outline_height


def read_name_file(
    name_path: Path,
    area_scale: Optional[float],
    coord_scale: float,
    rename: bool,
) -> NameFileData:
    raw_lines = name_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in raw_lines]
    lines = [ln for ln in lines if ln]

    raw_width, raw_height, outline_width, outline_height = parse_outline(lines, coord_scale)

    clusters: List[Cluster] = []
    i = 0
    used_names = set()

    # If area_scale is not given, keep area units consistent with coordinate scaling.
    # Example: area is in original coordinate units^2, coord_scale=100, so area_scale=10000.
    effective_area_scale = coord_scale * coord_scale if area_scale is None else area_scale

    while i < len(lines):
        if not lines[i].startswith("Cluster_Name:"):
            i += 1
            continue

        original_name = lines[i].split(":", 1)[1].strip()
        if i + 2 >= len(lines):
            raise ValueError(f"Broken cluster record near line {i + 1} in {name_path}")

        if not lines[i + 1].startswith("Cluster_num:"):
            raise ValueError(f"Expected Cluster_num after Cluster_Name near line {i + 2} in {name_path}")
        if not lines[i + 2].startswith("Cluster_area:"):
            raise ValueError(f"Expected Cluster_area after Cluster_num near line {i + 3} in {name_path}")

        cluster_num = int(float(lines[i + 1].split(":", 1)[1].strip()))
        area = float(lines[i + 2].split(":", 1)[1].strip())

        if area < 0:
            raise ValueError(f"Negative area for cluster {original_name}: {area}")

        if rename:
            placer_name = f"c{len(clusters)}"
        else:
            placer_name = sanitize_name(original_name)

        base_name = placer_name
        suffix = 1
        while placer_name in used_names:
            placer_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(placer_name)

        scaled_area = max(area * effective_area_scale, 1.0)
        side = max(1, int(math.ceil(math.sqrt(scaled_area))))

        clusters.append(
            Cluster(
                original_name=original_name,
                placer_name=placer_name,
                cluster_num=cluster_num,
                area=area,
                side=side,
            )
        )
        i += 3

    if not clusters:
        raise ValueError(f"No clusters found in {name_path}")

    return NameFileData(
        clusters=clusters,
        raw_width=raw_width,
        raw_height=raw_height,
        outline_width=outline_width,
        outline_height=outline_height,
    )


def read_net_file(
    net_path: Path,
    original_to_placer: Dict[str, str],
    dedupe_policy: str = "max",
    skip_unknown: bool = True,
) -> List[Edge]:
    """
    Read source adjacency list and return undirected weighted edges.

    dedupe_policy:
        max   : for A->B and B->A, keep max weight
        first : keep first seen weight
        sum   : sum all repeated directed entries; use only when input intentionally lists additive edges.
    """
    edge_weight: Dict[Tuple[str, str], float] = {}
    unknown_nodes = set()
    malformed_lines = 0

    with net_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("Net_"):
                continue
            if not line.startswith("source:"):
                continue

            tokens = line.split()
            if len(tokens) < 2:
                malformed_lines += 1
                continue

            src_original = tokens[1]
            if src_original not in original_to_placer:
                unknown_nodes.add(src_original)
                if skip_unknown:
                    continue

            if (len(tokens) - 2) % 2 != 0:
                malformed_lines += 1

            src = original_to_placer.get(src_original, sanitize_name(src_original))
            pair_tokens = tokens[2:]

            for j in range(0, len(pair_tokens) - 1, 2):
                dst_original = pair_tokens[j]
                weight_token = pair_tokens[j + 1]
                if not is_number(weight_token):
                    malformed_lines += 1
                    continue
                if dst_original not in original_to_placer:
                    unknown_nodes.add(dst_original)
                    if skip_unknown:
                        continue

                dst = original_to_placer.get(dst_original, sanitize_name(dst_original))
                if src == dst:
                    continue

                weight = float(weight_token)
                if weight <= 0:
                    continue

                u, v = sorted((src, dst))
                key = (u, v)

                if key not in edge_weight:
                    edge_weight[key] = weight
                else:
                    if dedupe_policy == "max":
                        edge_weight[key] = max(edge_weight[key], weight)
                    elif dedupe_policy == "first":
                        pass
                    elif dedupe_policy == "sum":
                        edge_weight[key] += weight
                    else:
                        raise ValueError(f"Unsupported dedupe policy: {dedupe_policy}")

    if malformed_lines:
        print(f"[WARN] {net_path}: skipped/partially parsed {malformed_lines} malformed source lines/pairs", file=sys.stderr)
    if unknown_nodes:
        sample = ", ".join(sorted(list(unknown_nodes))[:10])
        more = " ..." if len(unknown_nodes) > 10 else ""
        print(f"[WARN] {net_path}: {len(unknown_nodes)} nodes appear in .net but not .name: {sample}{more}", file=sys.stderr)

    edges = [Edge(u, v, w) for (u, v), w in sorted(edge_weight.items())]
    return edges


def net_repeat_count(weight: float, mode: str, weight_scale: float, max_repeat: Optional[int]) -> int:
    if mode == "single":
        repeat = 1
    elif mode == "repeat":
        repeat = int(round(weight * weight_scale))
        repeat = max(1, repeat)
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")

    if max_repeat is not None:
        repeat = min(repeat, max_repeat)
    return repeat


def estimate_num_nets_and_pins(
    edges: List[Edge],
    mode: str,
    weight_scale: float,
    max_repeat: Optional[int],
    include_terminal_net: bool,
    terminal_anchor: Optional[str] = None,
) -> Tuple[int, int]:
    num_nets = 1 if include_terminal_net else 0

    if include_terminal_net:
        num_pins = len(TERMINALS)
        if terminal_anchor is not None:
            num_pins += 1
    else:
        num_pins = 0

    for e in edges:
        repeat = net_repeat_count(e.weight, mode, weight_scale, max_repeat)
        num_nets += repeat
        num_pins += 2 * repeat

    return num_nets, num_pins


def compute_outline(
    clusters: List[Cluster],
    utilization: float,
    user_width: Optional[int],
    user_height: Optional[int],
) -> Tuple[int, int, bool]:
    """
    Return (width, height, fixed_outline).

    fixed_outline is True when width and height are both explicitly provided by user or .name.
    In fixed mode, initial packing will not enlarge the outline silently.
    """
    total_block_area = sum(c.side * c.side for c in clusters)
    if utilization <= 0 or utilization > 1:
        raise ValueError("utilization must be in (0, 1]")

    if user_width is not None and user_height is not None:
        return int(user_width), int(user_height), True

    target_area = total_block_area / utilization
    side = int(math.ceil(math.sqrt(target_area)))
    side = max(side, max(c.side for c in clusters) + 2)

    if user_width is not None:
        width = int(user_width)
        height = int(math.ceil(target_area / max(width, 1)))
        height = max(height, max(c.side for c in clusters) + 2)
        return width, height, False

    if user_height is not None:
        height = int(user_height)
        width = int(math.ceil(target_area / max(height, 1)))
        width = max(width, max(c.side for c in clusters) + 2)
        return width, height, False

    return side, side, False


def pack_initial_locations(
    clusters: List[Cluster],
    outline_width: int,
    outline_height: int,
    gap: int,
    order: str,
    allow_enlarge: bool,
) -> Tuple[Dict[str, Tuple[int, int]], int, int]:
    """Simple row packing. If allow_enlarge is False, keep the outline fixed."""
    if order == "area_desc":
        pack_list = sorted(clusters, key=lambda c: (c.side, c.area), reverse=True)
    elif order == "input":
        pack_list = list(clusters)
    else:
        raise ValueError(f"Unsupported placement order: {order}")

    width = int(outline_width)
    height = int(outline_height)
    gap = max(0, int(gap))

    for _attempt in range(100):
        x = 0
        y = 0
        row_h = 0
        pos: Dict[str, Tuple[int, int]] = {}
        failed = False

        for c in pack_list:
            if x > 0 and x + c.side > width:
                x = 0
                y += row_h + gap
                row_h = 0

            if y + c.side > height:
                failed = True
                break

            pos[c.placer_name] = (int(x), int(y))
            x += c.side + gap
            row_h = max(row_h, c.side)

        if not failed:
            return pos, width, height

        if not allow_enlarge:
            raise RuntimeError(
                f"Initial row packing failed inside fixed outline {outline_width} x {outline_height}. "
                f"Try increasing --coord-scale consistency, reducing --gap, or using a larger outline."
            )

        width = int(math.ceil(width * 1.2 + 1))
        height = int(math.ceil(height * 1.2 + 1))

    raise RuntimeError("Could not pack initial locations after repeatedly enlarging outline")


def write_blocks(path: Path, clusters: List[Cluster]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("UCSC blocks 1.0\n")
        f.write("# Created      : generated by convert_cluster_to_ucla_v2.py\n")
        f.write("# User         : cluster-to-placer conversion\n")
        f.write("# Platform     : generic\n")
        f.write("\n")
        f.write("NumSoftRectangularBlocks : 0\n")
        f.write(f"NumHardRectilinearBlocks : {len(clusters)}\n")
        f.write(f"NumTerminals : {len(TERMINALS)}\n")
        f.write("\n")

        for c in clusters:
            s = int(c.side)
            f.write(f"{c.placer_name} hardrectilinear 4 (0, 0) (0, {s}) ({s}, {s}) ({s}, 0)\n")

        f.write("\n")
        for terminal_name, _desc in TERMINALS:
            f.write(f"{terminal_name} terminal\n")


def write_nets(
    path: Path,
    edges: List[Edge],
    mode: str,
    weight_scale: float,
    max_repeat: Optional[int],
    include_terminal_net: bool,
    terminal_anchor: Optional[str] = None,
) -> Tuple[int, int]:
    num_nets, num_pins = estimate_num_nets_and_pins(
        edges,
        mode,
        weight_scale,
        max_repeat,
        include_terminal_net,
        terminal_anchor,
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("UCLA nets 1.0\n")
        f.write("# Created      : generated by convert_cluster_to_ucla_v2.py\n")
        f.write("# User         : cluster-to-placer conversion\n")
        f.write("# Platform     : generic\n")
        f.write("\n")
        f.write(f"NumNets : {num_nets}\n")
        f.write(f"NumPins : {num_pins}\n")

        # First net: connect four corner terminals and one real block.
        # This avoids empty pin-centroid initialization in Floorplan.
        if include_terminal_net:
            degree = len(TERMINALS)
            if terminal_anchor is not None:
                degree += 1

            f.write(f"NetDegree : {degree}\n")

            for terminal_name, _desc in TERMINALS:
                f.write(f"{terminal_name} B\n")

            if terminal_anchor is not None:
                f.write(f"{terminal_anchor} B\n")

        for e in edges:
            repeat = net_repeat_count(e.weight, mode, weight_scale, max_repeat)

            for _ in range(repeat):
                f.write("NetDegree : 2\n")
                f.write(f"{e.u} B\n")
                f.write(f"{e.v} B\n")

    return num_nets, num_pins


def write_pl(
    path: Path,
    clusters: List[Cluster],
    positions: Dict[str, Tuple[int, int]],
    outline_width: int,
    outline_height: int,
) -> None:
    terminal_pos = {
        "p1": (0, 0),
        "p2": (int(outline_width), 0),
        "p3": (0, int(outline_height)),
        "p4": (int(outline_width), int(outline_height)),
    }

    with path.open("w", encoding="utf-8") as f:
        f.write("UCSC blocks 1.0\n")
        f.write("# Created      : generated by convert_cluster_to_ucla_v2.py\n")
        f.write("# User         : cluster-to-placer conversion\n")
        f.write("# Platform     : generic\n")
        f.write("\n")

        for c in clusters:
            x, y = positions[c.placer_name]
            f.write(f"{c.placer_name}\t{random.randint(0, outline_width)}\t{random.randint(0, outline_height)}\n")

        f.write("\n")
        for terminal_name, _desc in TERMINALS:
            x, y = terminal_pos[terminal_name]
            f.write(f"{terminal_name}\t{x}\t{y}\n")


def write_map(path: Path, clusters: List[Cluster]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["placer_name", "original_cluster_name", "cluster_num", "original_area", "square_side", "scaled_square_area"])
        for c in clusters:
            writer.writerow([c.placer_name, c.original_name, c.cluster_num, c.area, c.side, c.side * c.side])


def convert_one_benchmark(
    name_path: Path,
    net_path: Path,
    output_dir: Path,
    benchmark: str,
    area_scale: Optional[float],
    coord_scale: float,
    weight_mode: str,
    weight_scale: float,
    max_repeat: Optional[int],
    utilization: float,
    outline_width: Optional[int],
    outline_height: Optional[int],
    gap: int,
    rename: bool,
    placement_order: str,
    dedupe_policy: str,
    include_terminal_net: bool,
) -> None:
    name_data = read_name_file(
        name_path,
        area_scale=area_scale,
        coord_scale=coord_scale,
        rename=rename,
    )
    clusters = name_data.clusters
    original_to_placer = {c.original_name: c.placer_name for c in clusters}
    edges = read_net_file(net_path, original_to_placer, dedupe_policy=dedupe_policy)

    # Priority: explicit CLI outline > Width/Height parsed from .name > automatic outline.
    width_hint = outline_width if outline_width is not None else name_data.outline_width
    height_hint = outline_height if outline_height is not None else name_data.outline_height

    width, height, fixed_outline = compute_outline(clusters, utilization, width_hint, height_hint)

    # Initialize all movable clusters at (0, 0).
    # Terminals are written separately by write_pl() and still use the four outline corners.
    positions = {
        c.placer_name: (0, 0)
        for c in clusters
    }
    final_width = int(width)
    final_height = int(height)

    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = output_dir / f"{benchmark}.blocks.txt"
    nets_path = output_dir / f"{benchmark}.nets.txt"
    pl_path = output_dir / f"{benchmark}.pl"
    map_path = output_dir / f"{benchmark}.cluster_map.csv"

    write_blocks(blocks_path, clusters)
    terminal_anchor = clusters[0].placer_name if include_terminal_net and len(clusters) > 0 else None

    num_nets, num_pins = write_nets(
        nets_path,
        edges,
        weight_mode,
        weight_scale,
        max_repeat,
        include_terminal_net,
        terminal_anchor,
    )
    write_pl(pl_path, clusters, positions, final_width, final_height)
    write_map(map_path, clusters)

    total_original_area = sum(c.area for c in clusters)
    total_scaled_square_area = sum(c.side * c.side for c in clusters)
    print("=" * 80)
    print(f"Benchmark       : {benchmark}")
    print(f"Input .name     : {name_path}")
    print(f"Input .net      : {net_path}")
    print(f"Clusters        : {len(clusters)}")
    print(f"Undirected edges: {len(edges)}")
    print(f"Terminal net    : {'yes' if include_terminal_net else 'no'}")
    print(f"Output nets     : {num_nets}")
    print(f"Output pins     : {num_pins}")
    if name_data.raw_width is not None or name_data.raw_height is not None:
        print(f"Name Width/Height: {name_data.raw_width} x {name_data.raw_height}")
    print(f"Coord scale     : {coord_scale}")
    print(f"Original area   : {total_original_area:.6f}")
    print(f"Scaled sq. area : {total_scaled_square_area}")
    print(f"Outline         : {final_width} x {final_height}")
    print(f"Wrote           : {blocks_path}")
    print(f"Wrote           : {nets_path}")
    print(f"Wrote           : {pl_path}")
    print(f"Wrote           : {map_path}")


def find_benchmarks(input_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for name_path in sorted(input_dir.glob("*.name")):
        benchmark = name_path.stem
        net_path = input_dir / f"{benchmark}.net"
        if not net_path.exists():
            print(f"[WARN] skip {benchmark}: missing {net_path.name}", file=sys.stderr)
            continue
        pairs.append((benchmark, name_path, net_path))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert cluster-level .name/.net files to UCLA/UCSC placer input files."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path, help="Directory containing many <benchmark>.name and <benchmark>.net pairs.")
    input_group.add_argument("--name-file", type=Path, help="Single .name file to convert. Must be used with --net-file.")

    parser.add_argument("--net-file", type=Path, help="Single .net file to convert. Required when --name-file is used.")
    parser.add_argument("--benchmark", type=str, help="Output benchmark name for single-file mode. Default: stem of --name-file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated .blocks.txt/.nets.txt/.pl files.")

    parser.add_argument("--coord-scale", type=float, default=100.0,
                        help="Scale Width/Height/coordinates to integers. Example: 22.95 with 100 -> 2295. Default: 100.")
    parser.add_argument("--area-scale", type=float, default=None,
                        help="Scale factor before square conversion: side=ceil(sqrt(area*area_scale)). "
                             "Default: coord_scale^2, keeping area consistent with coordinate scaling.")
    parser.add_argument("--weight-mode", choices=["repeat", "single"], default="repeat",
                        help="repeat: expand edge weight to multiple 2-pin nets; single: one 2-pin net per edge. Default: repeat")
    parser.add_argument("--weight-scale", type=float, default=1.0,
                        help="Scale factor for repeat count when --weight-mode repeat. Default: 1.0")
    parser.add_argument("--max-repeat", type=int, default=None,
                        help="Optional cap on repeated nets per edge to avoid huge .nets files.")
    parser.add_argument("--utilization", type=float, default=0.70,
                        help="Target utilization for automatically estimated outline if Width/Height are absent. Default: 0.70")
    parser.add_argument("--outline-width", type=int, default=None,
                        help="Optional fixed integer outline width. Overrides Width in .name.")
    parser.add_argument("--outline-height", type=int, default=None,
                        help="Optional fixed integer outline height. Overrides Height in .name.")
    parser.add_argument("--gap", type=int, default=1,
                        help="Gap between initially packed square blocks in .pl. Default: 1")
    parser.add_argument("--rename", action="store_true",
                        help="Rename clusters to c0,c1,... and write original names in .cluster_map.csv.")
    parser.add_argument("--placement-order", choices=["area_desc", "input"], default="area_desc",
                        help="Initial packing order in .pl. Default: area_desc")
    parser.add_argument("--dedupe-policy", choices=["max", "first", "sum"], default="max",
                        help="How to merge A->B and B->A duplicated adjacency entries. Default: max")
    parser.add_argument("--no-terminal-net", dest="include_terminal_net", action="store_false",
                        help="Disable the first terminal net connecting p1/p2/p3/p4.")
    parser.set_defaults(include_terminal_net=True)

    args = parser.parse_args()

    if args.name_file is not None and args.net_file is None:
        parser.error("--net-file is required when --name-file is used")
    if args.name_file is None and args.net_file is not None:
        parser.error("--net-file can only be used together with --name-file")
    if args.coord_scale <= 0:
        parser.error("--coord-scale must be positive")
    if args.area_scale is not None and args.area_scale <= 0:
        parser.error("--area-scale must be positive if provided")
    if args.weight_scale <= 0:
        parser.error("--weight-scale must be positive")
    if args.max_repeat is not None and args.max_repeat <= 0:
        parser.error("--max-repeat must be positive if provided")

    return args


def main() -> None:
    args = parse_args()

    if args.input_dir is not None:
        pairs = find_benchmarks(args.input_dir)
        if not pairs:
            raise FileNotFoundError(f"No <benchmark>.name/.net pairs found in {args.input_dir}")

        for benchmark, name_path, net_path in pairs:
            convert_one_benchmark(
                name_path=name_path,
                net_path=net_path,
                output_dir=args.output_dir / benchmark,
                benchmark=benchmark,
                area_scale=args.area_scale,
                coord_scale=args.coord_scale,
                weight_mode=args.weight_mode,
                weight_scale=args.weight_scale,
                max_repeat=args.max_repeat,
                utilization=args.utilization,
                outline_width=args.outline_width,
                outline_height=args.outline_height,
                gap=args.gap,
                rename=args.rename,
                placement_order=args.placement_order,
                dedupe_policy=args.dedupe_policy,
                include_terminal_net=args.include_terminal_net,
            )
    else:
        benchmark = args.benchmark if args.benchmark else args.name_file.stem
        convert_one_benchmark(
            name_path=args.name_file,
            net_path=args.net_file,
            output_dir=args.output_dir,
            benchmark=benchmark,
            area_scale=args.area_scale,
            coord_scale=args.coord_scale,
            weight_mode=args.weight_mode,
            weight_scale=args.weight_scale,
            max_repeat=args.max_repeat,
            utilization=args.utilization,
            outline_width=args.outline_width,
            outline_height=args.outline_height,
            gap=args.gap,
            rename=args.rename,
            placement_order=args.placement_order,
            dedupe_policy=args.dedupe_policy,
            include_terminal_net=args.include_terminal_net,
        )


if __name__ == "__main__":
    main()
