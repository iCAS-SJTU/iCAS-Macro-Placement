#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Bookshelf version of the AutoCluster-style partitionDesign flow.

This script is intentionally written to mirror the structure of the C++
AutoClusterMgr::partitionDesign() flow as much as possible, but without
OpenROAD/OpenDB/OpenSTA dependencies.

Input:\n  - either a benchmark .pt file using the Benchmark tensor schema\n  - or Bookshelf .nodes/.nets with optional .pl\n  - optional macro list for Bookshelf mode

Output, matching the spirit of the C++ flow:
  - <file_name>.block
  - <file_name>.net
  - <file_name>.weight
  - <file_name>.name
  - <file_name>.outdegree
  - <file_name>.mmdegree
  - <file_name>.mcdegree
  - <file_name>.mccdegree
  - cluster_graph.json
  - cell_to_cluster.txt

Main C++ correspondence:
  computeMetrics()             -> compute_metrics()
  createCluster()              -> create_initial_clusters()
  updateConnection()           -> update_connection()
  merge("top")                 -> merge_small_clusters()
  MLPart()                     -> split_large_cell_clusters()
  split macro/stdcell cluster  -> split_macro_std_clusters()
  mergeMacro()                 -> merge_macro_by_signature()
  MacroPart()                  -> macro_part_by_area()
  virtual_map_                 -> virtual_map dict
  output .block/.net/.name     -> write_autocluster_outputs()

Important simplification:
  - No OpenSTA timing graph is used.
  - findAdjacencies() is approximated by optional macro-cell-cell and
    cell-mediated macro-macro virtual connections derived from Bookshelf nets.
  - No real pin direction is used unless your .nets direction field exists;
    we still mostly treat connections undirected for clustering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional


# ============================================================
# Basic data structures
# ============================================================


@dataclass
class Inst:
    id: int
    name: str
    width: float
    height: float
    is_macro: bool = False
    is_terminal: bool = False
    x: float = 0.0
    y: float = 0.0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class PinRef:
    inst_id: int
    direction: str = "B"  # I/O/B/unknown. Bookshelf may provide I/O.


@dataclass
class Net:
    id: int
    name: str
    pins: List[PinRef] = field(default_factory=list)
    weight: int = 1

    def inst_ids(self) -> List[int]:
        seen = set()
        out = []
        for p in self.pins:
            if p.inst_id not in seen:
                out.append(p.inst_id)
                seen.add(p.inst_id)
        return out


@dataclass
class Cluster:
    id: int
    name: str
    inst_ids: List[int] = field(default_factory=list)
    output_connections: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    input_connections: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add_inst(self, inst_id: int) -> None:
        if inst_id not in self.inst_ids:
            self.inst_ids.append(inst_id)

    def remove_inst(self, inst_id: int) -> None:
        if inst_id in self.inst_ids:
            self.inst_ids.remove(inst_id)

    def add_output_connection(self, target_id: int, weight: int = 1) -> None:
        self.output_connections[target_id] += int(weight)

    def add_input_connection(self, src_id: int, weight: int = 1) -> None:
        self.input_connections[src_id] += int(weight)

    def init_connection(self) -> None:
        self.output_connections = defaultdict(int)
        self.input_connections = defaultdict(int)


# ============================================================
# Parser
# ============================================================


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_nodes(path: str) -> Tuple[List[Inst], Dict[str, int]]:
    insts: List[Inst] = []
    name_to_id: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue
            if line.startswith("UCLA") or line.startswith("NumNodes") or line.startswith("NumTerminals"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                w = float(parts[1])
                h = float(parts[2])
            except ValueError:
                continue
            name = parts[0]
            is_terminal = any(p.lower().startswith("terminal") for p in parts[3:])
            iid = len(insts)
            insts.append(Inst(iid, name, w, h, is_terminal=is_terminal))
            name_to_id[name] = iid
    return insts, name_to_id


def parse_pl(path: Optional[str], insts: List[Inst], name_to_id: Dict[str, int]) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 3 or parts[0] not in name_to_id:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            inst = insts[name_to_id[parts[0]]]
            inst.x = x
            inst.y = y


def parse_nets(path: str, name_to_id: Dict[str, int]) -> List[Net]:
    nets: List[Net] = []
    cur: Optional[Net] = None
    seen: Set[int] = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_comment(raw)
            if not line:
                continue
            if line.startswith("UCLA") or line.startswith("NumNets") or line.startswith("NumPins"):
                continue
            if line.startswith("NetDegree"):
                if cur is not None and len(cur.inst_ids()) >= 2:
                    nets.append(cur)
                parts = line.replace(":", " ").split()
                net_name = f"Net_{len(nets)}"
                for i, p in enumerate(parts):
                    if p.isdigit() and i + 1 < len(parts):
                        net_name = parts[i + 1]
                        break
                cur = Net(id=len(nets), name=net_name)
                seen = set()
                continue

            if cur is None:
                continue
            parts = line.split()
            if not parts:
                continue
            inst_name = parts[0]
            if inst_name not in name_to_id:
                continue
            iid = name_to_id[inst_name]
            if iid in seen:
                continue
            direction = "B"
            if len(parts) >= 2 and parts[1] in {"I", "O", "B", "INPUT", "OUTPUT"}:
                direction = parts[1]
            cur.pins.append(PinRef(iid, direction))
            seen.add(iid)

    if cur is not None and len(cur.inst_ids()) >= 2:
        nets.append(cur)
    for i, n in enumerate(nets):
        n.id = i
    return nets


def read_macro_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    s = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_comment(raw)
            if line:
                s.add(line.split()[0])
    return s


def mark_macros(insts: List[Inst], macro_names: Set[str], macro_area_ratio: float) -> None:
    areas = sorted(i.area for i in insts if i.area > 0)
    median_area = areas[len(areas) // 2] if areas else 1.0
    threshold = median_area * macro_area_ratio
    for inst in insts:
        lname = inst.name.lower()
        if inst.name in macro_names:
            inst.is_macro = True
        elif "macro" in lname or "sram" in lname or "ram" in lname:
            inst.is_macro = True
        elif inst.area >= threshold:
            inst.is_macro = True


# ============================================================
# AutoCluster-like manager
# ============================================================


class StandaloneAutoCluster:
    def __init__(
        self,
        insts: List[Inst],
        nets: List[Net],
        max_num_macro: int,
        min_num_macro: int,
        max_num_inst: int,
        min_num_inst: int,
        net_threshold: int,
        virtual_weight: int,
        ignore_net_threshold: int,
        max_net_degree: int,
        seed: int = 0,
    ) -> None:
        self.insts = insts
        self.nets = nets
        self.max_num_macro = max_num_macro
        self.min_num_macro = min_num_macro
        self.max_num_inst = max_num_inst
        self.min_num_inst = min_num_inst
        self.net_threshold = net_threshold
        self.virtual_weight = virtual_weight
        self.ignore_net_threshold = ignore_net_threshold
        self.max_net_degree = max_net_degree
        self.rng = random.Random(seed)

        self.cluster_id = 0
        self.cluster_map: Dict[int, Cluster] = {}
        self.cluster_list: List[Cluster] = []
        self.merge_cluster_list: List[Cluster] = []
        self.break_cluster_queue: deque[Cluster] = deque()
        self.mlpart_cluster_queue: deque[Cluster] = deque()
        self.inst_map: Dict[int, int] = {}  # inst_id -> cluster_id
        self.virtual_map: Dict[int, int] = {}  # C++ virtual_map_[target_id] = std_cell_id

        # degree statistics, matching the C++ output files
        self.macro_out_degree_macro: Dict[str, List[str]] = defaultdict(list)
        self.macro_out_degree_cell: Dict[str, List[str]] = defaultdict(list)
        self.macro_out_degree_cell_cell: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # ----------------------------- metrics -----------------------------

    def cluster_area(self, c: Cluster) -> float:
        return sum(self.insts[i].area for i in c.inst_ids)

    def cluster_num_macro(self, c: Cluster) -> int:
        return sum(1 for i in c.inst_ids if self.insts[i].is_macro or self.insts[i].is_terminal)

    def cluster_num_inst(self, c: Cluster) -> int:
        return sum(1 for i in c.inst_ids if not self.insts[i].is_macro and not self.insts[i].is_terminal)

    def compute_metrics(self) -> Tuple[int, int, float]:
        num_inst = sum(1 for i in self.insts if not i.is_macro and not i.is_terminal)
        num_macro = sum(1 for i in self.insts if i.is_macro or i.is_terminal)
        area = sum(i.area for i in self.insts)
        return num_inst, num_macro, area

    # ----------------------------- cluster creation -----------------------------

    def new_cluster(self, name: str, forced_id: Optional[int] = None) -> Cluster:
        if forced_id is None:
            self.cluster_id += 1
            cid = self.cluster_id
        else:
            cid = forced_id
        c = Cluster(cid, name)
        self.cluster_map[cid] = c
        return c

    def create_initial_clusters(self) -> None:
        """Standalone equivalent of createCluster().

        Since Bookshelf has no hierarchy, create one top cluster first.
        It will later be split by MLPart-style connectivity partitioning.
        """
        c = self.new_cluster("top_instance")
        for inst in self.insts:
            c.add_inst(inst.id)
            self.inst_map[inst.id] = c.id
        self.cluster_list.append(c)
        if self.cluster_num_inst(c) > self.max_num_inst or self.cluster_num_macro(c) > self.max_num_macro:
            self.break_cluster_queue.append(c)

    # ----------------------------- connection update -----------------------------

    def update_connection(self) -> None:
        for c in self.cluster_map.values():
            c.init_connection()
        self.calculate_connection()

    def calculate_connection(self) -> None:
        """Equivalent of calculateConnection(), but undirected/pairwise for Bookshelf.

        For each net, obtain unique clusters. Then add pairwise bidirectional
        connections. This replaces dbITerm direction-dependent driver/load logic.
        """
        for net in self.nets:
            pin_ids = net.inst_ids()
            if self.max_net_degree > 0 and len(pin_ids) > self.max_net_degree:
                continue
            cids = []
            seen = set()
            for iid in pin_ids:
                if iid not in self.inst_map:
                    continue
                cid = self.inst_map[iid]
                if cid not in seen:
                    seen.add(cid)
                    cids.append(cid)
            if len(cids) < 2:
                continue
            for u, v in combinations(sorted(cids), 2):
                self.cluster_map[u].add_output_connection(v, net.weight)
                self.cluster_map[v].add_input_connection(u, net.weight)
                self.cluster_map[v].add_output_connection(u, net.weight)
                self.cluster_map[u].add_input_connection(v, net.weight)

    # ----------------------------- merge small clusters -----------------------------

    def merge_cluster(self, src: Cluster, target: Cluster) -> None:
        for iid in list(target.inst_ids):
            src.add_inst(iid)
            self.inst_map[iid] = src.id
        self.cluster_map.pop(target.id, None)
        if target in self.cluster_list:
            self.cluster_list.remove(target)
        if target in self.merge_cluster_list:
            self.merge_cluster_list.remove(target)

    def merge_small_clusters(self, parent_name: str) -> None:
        """Equivalent of merge("top") / mergeUtil().

        Small clusters are merged based on similar connection signatures.
        """
        if not self.merge_cluster_list:
            return
        if len(self.merge_cluster_list) == 1:
            self.cluster_list.append(self.merge_cluster_list[0])
            self.merge_cluster_list.clear()
            self.update_connection()
            return

        outside_ids = [c.id for c in self.cluster_list if c not in self.merge_cluster_list]
        groups: Dict[Tuple[int, ...], List[Cluster]] = defaultdict(list)
        for c in self.merge_cluster_list:
            signature = []
            for oid in outside_ids:
                conn = c.input_connections.get(oid, 0) + c.output_connections.get(oid, 0)
                signature.append(1 if conn > self.net_threshold else 0)
            groups[tuple(signature)].append(c)

        merge_index = 0
        new_small = []
        for _, group in groups.items():
            base = group[0]
            for g in group[1:]:
                self.merge_cluster(base, g)
            if self.cluster_num_inst(base) >= self.min_num_inst or self.cluster_num_macro(base) >= self.min_num_macro:
                base.name = f"{parent_name}_cluster_{merge_index}"
                merge_index += 1
                if base not in self.cluster_list:
                    self.cluster_list.append(base)
            else:
                new_small.append(base)
        self.merge_cluster_list = new_small

        if len(self.merge_cluster_list) > 1:
            base = self.merge_cluster_list[0]
            for c in list(self.merge_cluster_list[1:]):
                self.merge_cluster(base, c)
            base.name = f"{parent_name}_cluster_{merge_index}"
            if base not in self.cluster_list:
                self.cluster_list.append(base)
            self.merge_cluster_list.clear()
        elif len(self.merge_cluster_list) == 1:
            base = self.merge_cluster_list[0]
            base.name = f"{parent_name}_cluster_{merge_index}"
            if base not in self.cluster_list:
                self.cluster_list.append(base)
            self.merge_cluster_list.clear()
        self.update_connection()

    # ----------------------------- MLPart approximation -----------------------------

    def split_large_cell_clusters(self) -> None:
        """Equivalent of repeatedly calling MLPart() on large flat clusters.

        Since we do not link MLPart, this uses a deterministic greedy 2-way split
        with area balance and connectivity seed expansion.
        """
        for c in list(self.cluster_list):
            if self.cluster_num_inst(c) > self.max_num_inst:
                self.mlpart_cluster_queue.append(c)

        while self.mlpart_cluster_queue:
            c = self.mlpart_cluster_queue.popleft()
            if c.id not in self.cluster_map:
                continue
            if self.cluster_num_inst(c) <= self.max_num_inst:
                continue
            self.mlpart(c)
            self.update_connection()

    def mlpart(self, c: Cluster) -> None:
        cell_ids = [i for i in c.inst_ids if not self.insts[i].is_macro and not self.insts[i].is_terminal]
        macro_ids = [i for i in c.inst_ids if self.insts[i].is_macro or self.insts[i].is_terminal]
        if len(cell_ids) < 2 * max(1, self.min_num_inst):
            return

        # Build local adjacency between cells.
        local_set = set(cell_ids)
        adj: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for net in self.nets:
            ids = [i for i in net.inst_ids() if i in local_set]
            if len(ids) < 2:
                continue
            if self.max_net_degree > 0 and len(ids) > self.max_net_degree:
                continue
            for u, v in combinations(ids, 2):
                adj[u][v] += 1
                adj[v][u] += 1

        # Pick two far-ish seeds: highest degree and least connected to it.
        degrees = [(sum(adj[i].values()), i) for i in cell_ids]
        degrees.sort(reverse=True)
        s0 = degrees[0][1]
        s1 = min(cell_ids, key=lambda x: adj[s0].get(x, 0) if x != s0 else 10**18)

        part0 = [s0]
        part1 = [s1]
        assigned = {s0: 0, s1: 1}
        area0 = self.insts[s0].area
        area1 = self.insts[s1].area
        target = sum(self.insts[i].area for i in cell_ids) / 2.0

        remaining = [i for i in cell_ids if i not in assigned]
        remaining.sort(key=lambda x: sum(adj[x].values()), reverse=True)
        for iid in remaining:
            conn0 = sum(adj[iid].get(j, 0) for j in part0)
            conn1 = sum(adj[iid].get(j, 0) for j in part1)
            if area0 > target * 1.05:
                p = 1
            elif area1 > target * 1.05:
                p = 0
            else:
                p = 0 if conn0 >= conn1 else 1
            if p == 0:
                part0.append(iid)
                area0 += self.insts[iid].area
                assigned[iid] = 0
            else:
                part1.append(iid)
                area1 += self.insts[iid].area
                assigned[iid] = 1

        # Remove old cluster.
        if c in self.cluster_list:
            self.cluster_list.remove(c)
        self.cluster_map.pop(c.id, None)

        c0 = self.new_cluster(c.name + "_cluster_0")
        c1 = self.new_cluster(c.name + "_cluster_1")
        for iid in part0:
            c0.add_inst(iid)
            self.inst_map[iid] = c0.id
        for iid in part1:
            c1.add_inst(iid)
            self.inst_map[iid] = c1.id

        # Keep macros with the more connected side.
        for mid in macro_ids:
            conn0 = sum(1 for net in self.nets if mid in net.inst_ids() and any(i in part0 for i in net.inst_ids()))
            conn1 = sum(1 for net in self.nets if mid in net.inst_ids() and any(i in part1 for i in net.inst_ids()))
            target_c = c0 if conn0 >= conn1 else c1
            target_c.add_inst(mid)
            self.inst_map[mid] = target_c.id

        self.cluster_list.extend([c0, c1])
        if self.cluster_num_inst(c0) > self.max_num_inst:
            self.mlpart_cluster_queue.append(c0)
        if self.cluster_num_inst(c1) > self.max_num_inst:
            self.mlpart_cluster_queue.append(c1)

    # ----------------------------- split macro/stdcell -----------------------------

    def split_macro_std_clusters(self) -> None:
        """Equivalent of C++ split: cluster_old -> _macro and _std_cell.

        C++ creates a negative-ID macro cluster, then later explodes it into
        per-macro clusters and uses virtual_map_ to remember its stdcell cluster.
        """
        par_cluster_vec = [c for c in list(self.cluster_list) if self.cluster_num_macro(c) > 0]

        for cluster_old in par_cluster_vec:
            macro_ids = [i for i in cluster_old.inst_ids if self.insts[i].is_macro or self.insts[i].is_terminal]
            if not macro_ids:
                continue

            neg_id = -cluster_old.id
            self.virtual_map[neg_id] = cluster_old.id
            macro_cluster = self.new_cluster(cluster_old.name + "_macro", forced_id=neg_id)
            for mid in macro_ids:
                macro_cluster.add_inst(mid)
                self.inst_map[mid] = neg_id
                cluster_old.remove_inst(mid)
            self.cluster_list.append(macro_cluster)
            cluster_old.name = cluster_old.name + "_std_cell"

        self.update_connection()

    # ----------------------------- mergeMacro -----------------------------

    def split_macros_to_singletons(self) -> None:
        """Force each macro/terminal to become its own cluster.

        This matches the requested behavior more closely than the original
        signature-based merge flow: every hard macro is isolated, while the
        std-cell clusters continue to be partitioned by the cell thresholds.
        """
        q = deque([c for c in list(self.cluster_list) if self.cluster_num_macro(c) > 0])
        while q:
            cluster_old = q.popleft()
            if cluster_old.id not in self.cluster_map:
                continue
            macro_ids = [i for i in cluster_old.inst_ids if self.insts[i].is_macro or self.insts[i].is_terminal]
            if not macro_ids:
                continue

            std_cell_id = self.virtual_map.get(cluster_old.id, 0)
            self.virtual_map.pop(cluster_old.id, None)
            self.cluster_map.pop(cluster_old.id, None)
            if cluster_old in self.cluster_list:
                self.cluster_list.remove(cluster_old)

            for mid in macro_ids:
                mc = self.new_cluster(self.insts[mid].name)
                mc.add_inst(mid)
                self.inst_map[mid] = mc.id
                self.cluster_list.append(mc)
                if std_cell_id:
                    self.virtual_map[mc.id] = std_cell_id

            self.update_connection()

    def merge_macro(self, parent_name: str, std_cell_id: int) -> None:
        if not self.merge_cluster_list:
            return
        if len(self.merge_cluster_list) == 1:
            self.virtual_map[self.merge_cluster_list[0].id] = std_cell_id
            self.cluster_list.append(self.merge_cluster_list[0])
            self.merge_cluster_list.clear()
            return
        self.merge_macro_util(parent_name, std_cell_id)

    def merge_macro_util(self, parent_name: str, std_cell_id: int) -> None:
        outside_ids = [c.id for c in self.cluster_list]
        groups: Dict[Tuple[int, ...], List[Cluster]] = defaultdict(list)
        for c in self.merge_cluster_list:
            signature = []
            for oid in outside_ids:
                conn = c.input_connections.get(oid, 0) + c.output_connections.get(oid, 0)
                signature.append(1 if conn > self.net_threshold else 0)
            groups[tuple(signature)].append(c)

        merge_index = 0
        self.merge_cluster_list.clear()
        for _, group in groups.items():
            base = group[0]
            for g in group[1:]:
                self.merge_cluster(base, g)
            base.name = f"{parent_name}_cluster_{merge_index}"
            merge_index += 1
            self.cluster_list.append(base)
            self.virtual_map[base.id] = std_cell_id

    # ----------------------------- MacroPart -----------------------------

    def macro_part_by_area(self) -> None:
        """Equivalent of MacroPart(): split macro clusters by identical area footprint."""
        q = deque([c for c in list(self.cluster_list) if self.cluster_num_macro(c) > self.min_num_macro])
        while q:
            old = q.popleft()
            if old.id not in self.cluster_map:
                continue
            macro_ids = [i for i in old.inst_ids if self.insts[i].is_macro or self.insts[i].is_terminal]
            groups: Dict[Tuple[float, float], List[int]] = defaultdict(list)
            for mid in macro_ids:
                inst = self.insts[mid]
                groups[(inst.width, inst.height)].append(mid)

            if old in self.cluster_list:
                self.cluster_list.remove(old)
            self.cluster_map.pop(old.id, None)
            old_virtual = self.virtual_map.pop(old.id, 0)

            part_id = 0
            new_ids = []
            for _, mids in groups.items():
                nc = self.new_cluster(f"{old.name}_part_{part_id}")
                part_id += 1
                for mid in mids:
                    nc.add_inst(mid)
                    self.inst_map[mid] = nc.id
                self.cluster_list.append(nc)
                new_ids.append(nc.id)
                if old_virtual:
                    self.virtual_map[nc.id] = old_virtual

            # C++ virtual_map_ between macro parts is quirky; here we preserve
            # the intent by adding virtual links later via add_virtual_weights().
        self.update_connection()

    # ----------------------------- virtual weights and timing approximation -----------------------------

    def add_virtual_weights(self) -> None:
        """Equivalent of the virtual_map_ loop in C++.

        In the pasted C++ these two lines are commented out. Here they are made
        optional through --enable-virtual-map-weight.
        """
        for target_id, std_cell_id in list(self.virtual_map.items()):
            if std_cell_id in self.cluster_map and target_id in self.cluster_map:
                self.cluster_map[std_cell_id].add_output_connection(target_id, self.virtual_weight)
                self.cluster_map[target_id].add_input_connection(std_cell_id, self.virtual_weight)
                self.cluster_map[target_id].add_output_connection(std_cell_id, self.virtual_weight)
                self.cluster_map[std_cell_id].add_input_connection(target_id, self.virtual_weight)

    def add_bookshelf_timing_like_virtual_edges(self, macro_cell_weight: int, cell_macro_weight: int, macro_macro_weight: int, macro_cell_cell_weight: int) -> None:
        """Standalone approximation of addTimingWeight() from the C++ code.

        It records:
          - macro_out_degree_macro
          - macro_out_degree_cell
          - macro_out_degree_cell_cell

        And adds:
          - macro -> cell cluster edges
          - cell -> macro cluster edges
          - macro -> macro virtual edges when two macros share a cell cluster
          - macro -> cell2 virtual edges for macro-cell-cell pattern
        """
        macro_cell_connect: Dict[int, int] = {}  # cell_cluster -> macro_cluster
        cell_macro_cluster: Dict[int, List[int]] = defaultdict(list)

        # First pass over existing real cluster-level connections.
        # We use output_connections as the current cluster graph.
        for src_id, src_cluster in list(self.cluster_map.items()):
            src_macro = self.cluster_num_macro(src_cluster) > 0
            src_cell = self.cluster_num_inst(src_cluster) > 0
            for sink_id, w in list(src_cluster.output_connections.items()):
                if sink_id not in self.cluster_map or sink_id == src_id:
                    continue
                sink_cluster = self.cluster_map[sink_id]
                sink_macro = self.cluster_num_macro(sink_cluster) > 0
                sink_cell = self.cluster_num_inst(sink_cluster) > 0

                if src_macro and sink_cell:
                    self.cluster_map[src_id].add_output_connection(sink_id, macro_cell_weight)
                    self.cluster_map[sink_id].add_input_connection(src_id, macro_cell_weight)
                    self.macro_out_degree_cell[src_cluster.name].append(sink_cluster.name)
                    macro_cell_connect[sink_id] = src_id

                elif src_cell and sink_macro:
                    self.cluster_map[src_id].add_output_connection(sink_id, cell_macro_weight)
                    self.cluster_map[sink_id].add_input_connection(src_id, cell_macro_weight)
                    cell_macro_cluster[src_id].append(sink_id)

                elif src_macro and sink_macro:
                    self.macro_out_degree_macro[src_cluster.name].append(sink_cluster.name)

                elif src_cell and sink_cell:
                    # macro-cell-cell: if src cell cluster was connected from a macro,
                    # add macro -> sink cell connection.
                    if src_id in macro_cell_connect:
                        macro_id = macro_cell_connect[src_id]
                        if macro_id in self.cluster_map and macro_id != sink_id:
                            macro_name = self.cluster_map[macro_id].name
                            self.cluster_map[macro_id].add_output_connection(sink_id, macro_cell_cell_weight)
                            self.cluster_map[sink_id].add_input_connection(macro_id, macro_cell_cell_weight)
                            self.macro_out_degree_cell_cell[macro_name].append((src_cluster.name, sink_cluster.name))

        # old indirect connection: cell cluster connected to multiple macros
        # => add macro-macro virtual edges.
        for cell_cluster, macros in cell_macro_cluster.items():
            unique_macros = sorted(set(macros))
            for m1, m2 in combinations(unique_macros, 2):
                if m1 in self.cluster_map and m2 in self.cluster_map:
                    self.cluster_map[m1].add_output_connection(m2, macro_macro_weight)
                    self.cluster_map[m2].add_input_connection(m1, macro_macro_weight)
                    self.cluster_map[m2].add_output_connection(m1, macro_macro_weight)
                    self.cluster_map[m1].add_input_connection(m2, macro_macro_weight)
                    self.macro_out_degree_macro[self.cluster_map[m1].name].append(self.cluster_map[m2].name)
                    self.macro_out_degree_macro[self.cluster_map[m2].name].append(self.cluster_map[m1].name)

    # ----------------------------- output -----------------------------

    def write_outputs(self, out_dir: str, file_name: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        id_inst_map: Dict[int, List[int]] = defaultdict(list)
        for iid, cid in self.inst_map.items():
            id_inst_map[cid].append(iid)

        # Derive actual macro degree views from the current cluster graph so the
        # degree files reflect real cluster-to-cluster connectivity even when the
        # timing-like virtual edge bookkeeping is disabled.
        actual_macro_to_macro: Dict[str, List[str]] = defaultdict(list)
        actual_macro_to_cell: Dict[str, List[str]] = defaultdict(list)
        for cid in sorted(self.cluster_map):
            c = self.cluster_map[cid]
            if self.cluster_num_macro(c) == 0:
                continue
            for tid, weight in sorted(c.output_connections.items()):
                if tid not in self.cluster_map or tid == cid or weight <= 0:
                    continue
                target = self.cluster_map[tid]
                if self.cluster_num_macro(target) > 0:
                    actual_macro_to_macro[c.name].append(target.name)
                elif self.cluster_num_inst(target) > 0:
                    actual_macro_to_cell[c.name].append(target.name)

        num_inst, num_macro, total_area = self.compute_metrics()

        # .block
        with open(os.path.join(out_dir, f"{file_name}.block"), "w", encoding="utf-8") as f:
            f.write(f"[INFO] Num clusters: {len(self.cluster_list)}\n")
            f.write("[INFO] Floorplan width: 0\n")
            f.write("[INFO] Floorplan height: 0\n")
            f.write("[INFO] Floorplan_lx: 0\n")
            f.write("[INFO] Floorplan_ly: 0\n")
            f.write(f"[INFO] Num std cells: {num_inst}\n")
            f.write(f"[INFO] Num macros: {num_macro}\n")
            f.write(f"[INFO] Total area: {total_area}\n")
            f.write("[INFO] Num buffers: 0\n")
            f.write("[INFO] Buffer area: 0\n\n")
            for cid in sorted(self.cluster_map):
                c = self.cluster_map[cid]
                area = self.cluster_area(c)
                if area == 0:
                    continue
                f.write(f"cluster: {c.name}\n")
                f.write(f"area:  {area}\n")
                if self.cluster_num_macro(c) > 0:
                    for iid in c.inst_ids:
                        inst = self.insts[iid]
                        if inst.is_macro or inst.is_terminal:
                            f.write(f"{inst.name}  {inst.width}   {inst.height}\n")
                f.write("\n")

        # .net
        with open(os.path.join(out_dir, f"{file_name}.net"), "w", encoding="utf-8") as f:
            net_id = 0
            for cid in sorted(self.cluster_map):
                c = self.cluster_map[cid]
                conn = c.output_connections
                non_self = [(tid, w) for tid, w in conn.items() if tid != cid and tid in self.cluster_map]
                if not non_self:
                    continue
                net_id += 1
                f.write(f"Net_{net_id}:  \n")
                f.write(f"source: {c.name}   ")
                for tid, weight in sorted(non_self):
                    out_w = weight if weight >= self.ignore_net_threshold else 0
                    f.write(f"{self.cluster_map[tid].name}   {out_w}   ")
                f.write("\n")
            f.write("\n")

        # .weight
        with open(os.path.join(out_dir, f"{file_name}.weight"), "w", encoding="utf-8") as f:
            for net in self.nets:
                pin_ids = net.inst_ids()
                if self.max_net_degree > 0 and len(pin_ids) > self.max_net_degree:
                    weight = 0
                else:
                    cids = []
                    seen = set()
                    for iid in pin_ids:
                        if iid not in self.inst_map:
                            continue
                        cid = self.inst_map[iid]
                        if cid not in seen:
                            seen.add(cid)
                            cids.append(cid)

                    weight = 0
                    if len(cids) >= 2:
                        for u, v in combinations(sorted(cids), 2):
                            w_uv = self.cluster_map[u].output_connections.get(v, 0)
                            w_vu = self.cluster_map[v].output_connections.get(u, 0)
                            weight += max(w_uv, w_vu)

                f.write(f"{net.name} {weight}\n")

        # .name
        with open(os.path.join(out_dir, f"{file_name}.name"), "w", encoding="utf-8") as f:
            for cid in sorted(self.cluster_map):
                c = self.cluster_map[cid]
                f.write(f"Cluster_Name: {c.name}\n")
                if self.cluster_num_macro(c) > 0:
                    f.write(f"Cluster_num: {self.cluster_num_macro(c)}\n")
                else:
                    f.write(f"Cluster_num: {self.cluster_num_inst(c)}\n")
                f.write(f"Cluster_area: {self.cluster_area(c)}\n")
            f.write("\n")

        # .outdegree
        with open(os.path.join(out_dir, f"{file_name}.outdegree"), "w", encoding="utf-8") as f:
            for cid in sorted(self.cluster_map):
                c = self.cluster_map[cid]
                if self.cluster_num_macro(c) > 0:
                    mm = len(actual_macro_to_macro[c.name])
                    mc = len(actual_macro_to_cell[c.name])
                    mcc = len(self.macro_out_degree_cell_cell[c.name])
                    f.write(f"name: {c.name} out degree: {mm + mc + mcc}\n")
                    f.write(f"macro_macro_out_degree = {mm}\n")
                    f.write(f"macro_cell_out_degree = {mc}\n")
                    f.write(f"macro_out_degree_cell_cell = {mcc}\n")
            f.write("\n")

        # .mmdegree
        with open(os.path.join(out_dir, f"{file_name}.mmdegree"), "w", encoding="utf-8") as f:
            for name in sorted(actual_macro_to_macro):
                values = actual_macro_to_macro[name]
                f.write(f"{name} {len(values)}\n")
                for v in values:
                    f.write(v + "\n")
                f.write("\n")
            f.write("\n")

        # .mcdegree
        with open(os.path.join(out_dir, f"{file_name}.mcdegree"), "w", encoding="utf-8") as f:
            for name in sorted(actual_macro_to_cell):
                values = actual_macro_to_cell[name]
                f.write(f"{name} {len(values)}\n")
                for v in values:
                    f.write(v + "\n")
                f.write("\n")
            f.write("\n")

        # .mccdegree
        with open(os.path.join(out_dir, f"{file_name}.mccdegree"), "w", encoding="utf-8") as f:
            for name, values in self.macro_out_degree_cell_cell.items():
                f.write(f"{name} {len(values)}\n")
                for a, b in values:
                    f.write(f"{a} {b}\n")
                f.write("\n")
            f.write("\n")

        # cell_to_cluster.txt
        with open(os.path.join(out_dir, "cell_to_cluster.txt"), "w", encoding="utf-8") as f:
            for iid, cid in sorted(self.inst_map.items()):
                if cid in self.cluster_map:
                    f.write(f"{self.insts[iid].name} {self.cluster_map[cid].name}\n")

        # JSON
        data = {
            "clusters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "area": self.cluster_area(c),
                    "num_cell": self.cluster_num_inst(c),
                    "num_macro": self.cluster_num_macro(c),
                    "members": [
                        {
                            "name": self.insts[iid].name,
                            "type": "macro" if self.insts[iid].is_macro else ("terminal" if self.insts[iid].is_terminal else "cell"),
                            "width": self.insts[iid].width,
                            "height": self.insts[iid].height,
                            "area": self.insts[iid].area,
                            "x": self.insts[iid].x,
                            "y": self.insts[iid].y,
                        }
                        for iid in c.inst_ids
                    ],
                    "connections": [
                        {"target": tid, "target_name": self.cluster_map[tid].name, "weight": w}
                        for tid, w in sorted(c.output_connections.items())
                        if tid in self.cluster_map and tid != c.id
                    ],
                }
                for c in sorted(self.cluster_map.values(), key=lambda x: x.id)
            ]
        }
        with open(os.path.join(out_dir, "cluster_graph.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ----------------------------- full flow -----------------------------

    def partition_design(self, enable_virtual_map_weight: bool, enable_timing_like_virtual: bool, macro_cell_weight: int, cell_macro_weight: int, macro_macro_weight: int, macro_cell_cell_weight: int) -> None:
        print("Running Partition Design...")
        print("max_num_macro", self.max_num_macro)

        num_inst, num_macro, area = self.compute_metrics()
        print("Traversed logical hierarchy")
        print("\tNumber of std cell instances:", num_inst)
        print("\tTotal area:", area)
        print("\tNumber of hard macros:", num_macro)

        # createCluster(cluster_id)
        self.create_initial_clusters()
        self.update_connection()

        # merge("top")
        self.merge_small_clusters("top")

        # breakCluster loop is omitted because Bookshelf has no hierarchy.

        # MLPart loop
        self.split_large_cell_clusters()

        # split macro/stdcell clusters
        self.split_macro_std_clusters()

        # Force every macro to become its own cluster.
        self.split_macros_to_singletons()

        # updateConnection()
        self.update_connection()

        # virtual_map_ loop
        if enable_virtual_map_weight:
            self.add_virtual_weights()

        # findAdjacencies() approximation
        if enable_timing_like_virtual:
            self.add_bookshelf_timing_like_virtual_edges(
                macro_cell_weight=macro_cell_weight,
                cell_macro_weight=cell_macro_weight,
                macro_macro_weight=macro_macro_weight,
                macro_cell_cell_weight=macro_cell_cell_weight,
            )


# ============================================================
# CLI
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bookshelf standalone AutoCluster-like partitionDesign")
    p.add_argument("--pt", default=None, help="Input Benchmark .pt file. If provided, .nodes/.nets are not needed.")
    p.add_argument(
        "--benchmark-dir",
        default=None,
        help="Competition-style benchmark directory, e.g. external/MacroPlacement/Testcases/ICCAD04/ibm01",
    )
    p.add_argument("--plc-netlist", default=None, help="MacroPlacement netlist.pb.txt input")
    p.add_argument("--plc-initial", default=None, help="Optional initial.plc paired with --plc-netlist")
    p.add_argument("--nodes", default=None, help="Bookshelf .nodes file, used when --pt is not provided")
    p.add_argument("--nets", default=None, help="Bookshelf .nets file, used when --pt is not provided")
    p.add_argument("--pl", default=None, help="Optional Bookshelf .pl file")
    p.add_argument("--out", required=True)
    p.add_argument("--file-name", default="partition")
    p.add_argument("--macro-list", default=None)
    p.add_argument("--macro-area-ratio", type=float, default=10.0)

    # C++ style parameters
    p.add_argument("--max-num-macro", type=int, default=10)
    p.add_argument("--min-num-macro", type=int, default=1)
    p.add_argument("--max-num-inst", type=int, default=5000)
    p.add_argument("--min-num-inst", type=int, default=50)
    p.add_argument("--net-threshold", type=int, default=0)
    p.add_argument("--virtual-weight", type=int, default=10)
    p.add_argument("--ignore-net-threshold", type=int, default=0)
    p.add_argument("--max-net-degree", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    # The C++ pasted virtual_map loop is commented. This option lets you enable it.
    p.add_argument("--enable-virtual-map-weight", action="store_true")

    # Standalone approximation of addTimingWeight() virtual edges.
    p.add_argument("--enable-timing-like-virtual", action="store_true")
    p.add_argument("--macro-cell-weight", type=int, default=4)
    p.add_argument("--cell-macro-weight", type=int, default=1)
    p.add_argument("--macro-macro-weight", type=int, default=2)
    p.add_argument("--macro-cell-cell-weight", type=int, default=2)
    return p


def load_pt_benchmark(path: str) -> Tuple[List[Inst], List[Net]]:
    import torch

    data = torch.load(path, weights_only=False)

    name = data.get("name", "benchmark")
    num_macros = int(data["num_macros"])
    num_hard = int(data.get("num_hard_macros", num_macros))
    macro_positions = data["macro_positions"]
    macro_sizes = data["macro_sizes"]
    macro_fixed = data.get("macro_fixed", torch.zeros(num_macros, dtype=torch.bool))
    macro_names = data.get("macro_names", [f"macro_{i}" for i in range(num_macros)])

    insts: List[Inst] = []
    for i in range(num_macros):
        w = float(macro_sizes[i, 0].item())
        h = float(macro_sizes[i, 1].item())
        x = float(macro_positions[i, 0].item())
        y = float(macro_positions[i, 1].item())
        insts.append(
            Inst(
                id=i,
                name=str(macro_names[i]),
                width=w,
                height=h,
                is_macro=(i < num_hard),
                is_terminal=bool(macro_fixed[i].item()) if hasattr(macro_fixed[i], "item") else bool(macro_fixed[i]),
                x=x,
                y=y,
            )
        )

    raw_net_nodes = data["net_nodes"]
    raw_net_weights = data.get("net_weights", torch.ones(len(raw_net_nodes)))
    nets: List[Net] = []
    for ni, nodes in enumerate(raw_net_nodes):
        pins: List[PinRef] = []
        seen = set()
        for v in nodes.tolist():
            iid = int(v)
            if 0 <= iid < num_macros and iid not in seen:
                pins.append(PinRef(iid, "B"))
                seen.add(iid)
        if len(pins) >= 2:
            weight = int(round(float(raw_net_weights[ni].item()))) if ni < len(raw_net_weights) else 1
            weight = max(1, weight)
            nets.append(Net(id=len(nets), name=f"net_{ni}", pins=pins, weight=weight))

    print(f"[INFO] Loaded PT benchmark: {name}")
    print(f"[INFO] num_macros={num_macros}, num_hard_macros={num_hard}, num_soft_macros={num_macros - num_hard}")
    print(f"[INFO] usable nets={len(nets)}")
    return insts, nets


def load_plc_benchmark(netlist_path: str, plc_path: Optional[str]) -> Tuple[List[Inst], List[Net]]:
    import sys

    plc_client_dir = "/Users/tttiko/Desktop/partcl-competition/MacroPlacement-shallow/CodeElements/Plc_client"
    if plc_client_dir not in sys.path:
        sys.path.insert(0, plc_client_dir)

    from plc_client_os import PlacementCost

    plc = PlacementCost(netlist_path)
    if plc_path:
        plc.restore_placement(plc_path, ifInital=True, ifReadComment=True)

    insts: List[Inst] = []
    plc_idx_to_inst_id: Dict[int, int] = {}

    component_indices = list(plc.hard_macro_indices) + list(plc.soft_macro_indices)
    for plc_idx in component_indices:
        node = plc.modules_w_pins[plc_idx]
        x, y = node.get_pos()
        inst_id = len(insts)
        plc_idx_to_inst_id[plc_idx] = inst_id
        insts.append(
            Inst(
                id=inst_id,
                name=node.get_name(),
                width=float(node.get_width()),
                height=float(node.get_height()),
                is_macro=(plc_idx in plc.hard_macro_indices),
                is_terminal=bool(node.get_fix_flag()),
                x=float(x),
                y=float(y),
            )
        )

    nets: List[Net] = []
    for net_name, sinks in plc.nets.items():
        pin_names = [net_name] + list(sinks)
        seen: Set[int] = set()
        pins: List[PinRef] = []
        for pin_name in pin_names:
            parent = pin_name.split("/", 1)[0]
            for plc_idx, inst_id in plc_idx_to_inst_id.items():
                node = plc.modules_w_pins[plc_idx]
                if node.get_name() == parent and inst_id not in seen:
                    pins.append(PinRef(inst_id, "B"))
                    seen.add(inst_id)
                    break
        if len(pins) >= 2:
            nets.append(Net(id=len(nets), name=f"net_{len(nets)}", pins=pins, weight=1))

    print(f"[INFO] Loaded PLC benchmark from {netlist_path}")
    print(
        f"[INFO] hard={len(plc.hard_macro_indices)}, soft={len(plc.soft_macro_indices)}, "
        f"ports={len(plc.port_indices)}, usable nets={len(nets)}"
    )
    return insts, nets


def load_benchmark_dir(benchmark_dir: str) -> Tuple[List[Inst], List[Net]]:
    netlist_path = os.path.join(benchmark_dir, "netlist.pb.txt")
    plc_path = os.path.join(benchmark_dir, "initial.plc")
    return load_plc_benchmark(netlist_path, plc_path)


def main() -> None:
    args = build_argparser().parse_args()

    if args.pt:
        insts, nets = load_pt_benchmark(args.pt)
    elif args.benchmark_dir:
        insts, nets = load_benchmark_dir(args.benchmark_dir)
    elif args.plc_netlist:
        insts, nets = load_plc_benchmark(args.plc_netlist, args.plc_initial)
    else:
        if args.nodes is None or args.nets is None:
            raise ValueError(
                "Provide one of: --pt, --benchmark-dir, --plc-netlist, or both --nodes and --nets."
            )
        insts, name_to_id = parse_nodes(args.nodes)
        parse_pl(args.pl, insts, name_to_id)
        macros = read_macro_list(args.macro_list)
        mark_macros(insts, macros, args.macro_area_ratio)
        nets = parse_nets(args.nets, name_to_id)

    mgr = StandaloneAutoCluster(
        insts=insts,
        nets=nets,
        max_num_macro=args.max_num_macro,
        min_num_macro=args.min_num_macro,
        max_num_inst=args.max_num_inst,
        min_num_inst=args.min_num_inst,
        net_threshold=args.net_threshold,
        virtual_weight=args.virtual_weight,
        ignore_net_threshold=args.ignore_net_threshold,
        max_net_degree=args.max_net_degree,
        seed=args.seed,
    )
    mgr.partition_design(
        enable_virtual_map_weight=args.enable_virtual_map_weight,
        enable_timing_like_virtual=args.enable_timing_like_virtual,
        macro_cell_weight=args.macro_cell_weight,
        cell_macro_weight=args.cell_macro_weight,
        macro_macro_weight=args.macro_macro_weight,
        macro_cell_cell_weight=args.macro_cell_cell_weight,
    )
    mgr.write_outputs(args.out, args.file_name)
    print("[DONE] output:", args.out)



def cluster(
    pt=None,
    benchmark_dir=None,
    plc_netlist=None,
    plc_initial=None,
    nodes=None,
    nets=None,
    pl=None,
    out=None,
    file_name="partition",
    macro_list=None,
    macro_area_ratio=10.0,

    # C++ style parameters
    max_num_macro=10,
    min_num_macro=1,
    max_num_inst=5000,
    min_num_inst=50,
    net_threshold=0,
    virtual_weight=10,
    ignore_net_threshold=0,
    max_net_degree=100,
    seed=0,

    # virtual map / timing-like virtual edges
    enable_virtual_map_weight=False,
    enable_timing_like_virtual=False,
    macro_cell_weight=4,
    cell_macro_weight=1,
    macro_macro_weight=2,
    macro_cell_cell_weight=2,
) -> None:
    """
    Python function interface for standalone AutoCluster-style partitioning.

    Supported input modes:
      1. pt benchmark:
         cluster(pt="benchmark.pt", out="./cluster_out")

      2. benchmark directory:
         cluster(benchmark_dir="xxx/ibm01", out="./cluster_out")

      3. PLC netlist:
         cluster(plc_netlist="netlist.pb.txt", plc_initial="initial.plc", out="./cluster_out")

      4. Bookshelf:
         cluster(nodes="xxx.nodes", nets="xxx.nets", pl="xxx.pl", out="./cluster_out")
    """

    if out is None:
        raise ValueError("out must be provided.")

    # ------------------------------------------------------------
    # Load benchmark
    # ------------------------------------------------------------
    if pt:
        insts, nets_data = load_pt_benchmark(pt)

    elif benchmark_dir:
        insts, nets_data = load_benchmark_dir(benchmark_dir)

    elif plc_netlist:
        insts, nets_data = load_plc_benchmark(
            plc_netlist,
            plc_initial,
        )

    else:
        if nodes is None or nets is None:
            raise ValueError(
                "Provide one of: pt, benchmark_dir, plc_netlist, "
                "or both nodes and nets."
            )

        insts, name_to_id = parse_nodes(nodes)

        parse_pl(
            pl,
            insts,
            name_to_id,
        )

        macros = read_macro_list(macro_list)

        mark_macros(
            insts,
            macros,
            macro_area_ratio,
        )

        nets_data = parse_nets(
            nets,
            name_to_id,
        )

    # ------------------------------------------------------------
    # Build standalone AutoCluster manager
    # ------------------------------------------------------------
    mgr = StandaloneAutoCluster(
        insts=insts,
        nets=nets_data,
        max_num_macro=max_num_macro,
        min_num_macro=min_num_macro,
        max_num_inst=max_num_inst,
        min_num_inst=min_num_inst,
        net_threshold=net_threshold,
        virtual_weight=virtual_weight,
        ignore_net_threshold=ignore_net_threshold,
        max_net_degree=max_net_degree,
        seed=seed,
    )

    # ------------------------------------------------------------
    # Run partitionDesign-like flow
    # ------------------------------------------------------------
    mgr.partition_design(
        enable_virtual_map_weight=enable_virtual_map_weight,
        enable_timing_like_virtual=enable_timing_like_virtual,
        macro_cell_weight=macro_cell_weight,
        cell_macro_weight=cell_macro_weight,
        macro_macro_weight=macro_macro_weight,
        macro_cell_cell_weight=macro_cell_cell_weight,
    )

    # ------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------
    mgr.write_outputs(
        out,
        file_name,
    )

    print("[DONE] output:", out)

if __name__ == "__main__":
    main()




"""
python bookshelf_cluster_partition.py \
  --pt benchmark.pt \
  --out ./cluster_out \
  --file-name result \
  --max-num-inst 5000 \
  --enable-virtual-map-weight \
  --enable-timing-like-virtual
"""
