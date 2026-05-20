from pathlib import Path
import time
import traceback
import csv
import logging
import sys
import os
import re
import shutil
from decimal import Decimal, InvalidOperation

import torch

# ============================================================
# Local third-party library paths
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
ICAS_ROOT = THIS_DIR / "icas_placer"

DREAMPLACE_INSTALL = ICAS_ROOT / "DREAMPlace" / "install"
DREAMPLACE_PKG = DREAMPLACE_INSTALL / "dreamplace"
CT_ROOT = ICAS_ROOT / "circuit_training"

for _p in (ICAS_ROOT, DREAMPLACE_PKG, DREAMPLACE_INSTALL, CT_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bookshelf_cluster_partition import cluster
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

from dreamplace import Params
import dreamplace.NonLinearPlace as NonLinearPlace

from direct_plc_placedb import (
    build_direct_placedb_from_plc,
    debug_direct_placedb_hpwl,
    extract_macro_placement_from_placedb,
    apply_net_weights_from_file,
)

RUN_JSON_DIR = Path(
    os.environ.get(
        "MPC_RUN_JSON_DIR",
        "/run_json",
    )
)

_LOG_FILE_HANDLE = None

class Placer:
    """
    Single-file DREAMPlace-based placer.

    Evaluator entry:
        placer = Placer()
        placement = placer.place(benchmark)

    The returned placement is a torch.Tensor with shape [num_macros, 2].
    """

    def __init__(self):
        self.verbose = self._bool_env("MPC_VERBOSE", False)
        self.auto_cluster = self._bool_env("MPC_AUTO_CLUSTER", False)
        self.run_json_dir = Path(
            os.environ.get(
                "MPC_RUN_JSON_DIR",
                "/run_json",
            )
        )
        self._log_file_handle = None

    def place(self, benchmark):
        try:
            placement = self.run_placement(
                json_file=None,
                benchmark=benchmark,
                verbose=self.verbose,
                auto_cluster=self.auto_cluster,
            )
        except Exception as e:
            print(f"[WARN] DREAMPlace flow failed, use greedy fallback: {repr(e)}", flush=True)
            placement = self.greedy_row_fallback(benchmark)

        placement = self._restore_fixed_macros(placement, benchmark)
        return placement


    def dreamplace_interface(self, params, placedb, benchmark=None, raw_plc=None, run_place=True, return_macro_placement=False):
        """
        Local DREAMPlace interface.

        This avoids importing dreamplace.Placer.dreamplace_interface,
        so we can use DREAMPlace as an external library.

        Args:
            params:
                DREAMPlace Params object.

            placedb:
                Converted DREAMPlace PlaceDB object.

            benchmark:
                Macro-place benchmark object. Required if return_macro_placement=True.

            raw_plc:
                Raw PLC object. Required if return_macro_placement=True.

            run_place:
                If False, only initialize NonLinearPlace and return placedb.

            return_macro_placement:
                If True, directly return macro placement tensor.

        Returns:
            If return_macro_placement=False:
                placedb, metrics

            If return_macro_placement=True:
                placement, placedb, metrics
        """
        params.printWelcome()
        if placedb is None:
            raise ValueError('placedb must be provided.')
        timer = None
        tt = time.time()
        placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
        logging.info('non-linear placement initialization takes %.2f seconds' % (time.time() - tt))
        metrics = None
        if run_place:
            metrics = placer(params, placedb)
            logging.info('non-linear placement takes %.2f seconds' % (time.time() - tt))
        if return_macro_placement:
            if benchmark is None:
                raise ValueError('benchmark must be provided when return_macro_placement=True.')
            if raw_plc is None:
                raise ValueError('raw_plc must be provided when return_macro_placement=True.')
            placement = extract_macro_placement_from_placedb(placedb=placedb, benchmark=benchmark, raw_plc=raw_plc, params=params)
            fixed_mask = benchmark.macro_fixed
            placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
            return (placement, placedb, metrics)
        return (placedb, metrics)

    def final_rescue_remaining_overlaps(self, placement, benchmark, grid_div=600, top_k_grid=300000, edge_top_k=1200, gap=0.001, verbose=True):
        """
        Final rescue legalizer for remaining few official hard-macro overlaps.

        This is intended for cases like:
            final violations: ['Macros 36 and 348 overlap']

        Strategy:
            - Only process remaining official overlap pairs.
            - Fixed macros never move.
            - If one side is fixed, only move the other side.
            - If both are movable, try both and choose the first legal global empty slot.
            - Candidate search is much stronger than the normal fast legalizer.
        """
        import numpy as np
        import torch
        import time
        t0 = time.time()
        input_is_torch = isinstance(placement, torch.Tensor)
        if input_is_torch:
            device = placement.device
            dtype = placement.dtype
            place_np = placement.detach().cpu().numpy().copy()
        else:
            device = None
            dtype = None
            place_np = np.asarray(placement, dtype=np.float64).copy()
        macro_sizes = benchmark.macro_sizes
        macro_fixed = benchmark.macro_fixed
        macro_positions = benchmark.macro_positions
        if isinstance(macro_sizes, torch.Tensor):
            macro_sizes_np = macro_sizes.detach().cpu().numpy()
        else:
            macro_sizes_np = np.asarray(macro_sizes)
        if isinstance(macro_fixed, torch.Tensor):
            macro_fixed_np = macro_fixed.detach().cpu().numpy().astype(bool)
        else:
            macro_fixed_np = np.asarray(macro_fixed).astype(bool)
        if isinstance(macro_positions, torch.Tensor):
            macro_positions_np = macro_positions.detach().cpu().numpy()
        else:
            macro_positions_np = np.asarray(macro_positions)
        place_np = np.asarray(place_np, dtype=np.float64)
        original_np = place_np.copy()
        macro_sizes_np = np.asarray(macro_sizes_np, dtype=np.float64)
        macro_positions_np = np.asarray(macro_positions_np, dtype=np.float64)
        num_hard = int(benchmark.num_hard_macros)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        w_arr = macro_sizes_np[:, 0]
        h_arr = macro_sizes_np[:, 1]
        left = right = bottom = top = None

        def to_output(arr):
            if input_is_torch:
                return torch.tensor(arr, dtype=dtype, device=device)
            return arr

        def update_rects():
            nonlocal left, right, bottom, top
            left = place_np[:, 0] - w_arr / 2.0
            right = place_np[:, 0] + w_arr / 2.0
            bottom = place_np[:, 1] - h_arr / 2.0
            top = place_np[:, 1] + h_arr / 2.0

        def restore_fixed():
            if macro_fixed_np.any():
                place_np[macro_fixed_np] = macro_positions_np[macro_fixed_np]
            update_rects()

        def clamp_center(i, cx, cy):
            wi = w_arr[i]
            hi = h_arr[i]
            cx = min(max(float(cx), wi / 2.0 + gap), canvas_w - wi / 2.0 - gap)
            cy = min(max(float(cy), hi / 2.0 + gap), canvas_h - hi / 2.0 - gap)
            return (cx, cy)

        def is_legal_hard_position(i, cx, cy):
            wi = w_arr[i]
            hi = h_arr[i]
            lx = cx - wi / 2.0
            ux = cx + wi / 2.0
            ly = cy - hi / 2.0
            uy = cy + hi / 2.0
            if lx < 0.0 or ly < 0.0 or ux > canvas_w or (uy > canvas_h):
                return False
            separated = (lx >= right[:num_hard]) | (ux <= left[:num_hard]) | (ly >= top[:num_hard]) | (uy <= bottom[:num_hard])
            overlap = ~separated
            overlap[i] = False
            return not bool(np.any(overlap))

        def move_order_for_pair(i, j):
            fixed_i = bool(macro_fixed_np[i])
            fixed_j = bool(macro_fixed_np[j])
            if fixed_i and fixed_j:
                return []
            if fixed_i and (not fixed_j):
                return [j]
            if fixed_j and (not fixed_i):
                return [i]
            area_i = w_arr[i] * h_arr[i]
            area_j = w_arr[j] * h_arr[j]
            if area_i <= area_j:
                return [i, j]
            return [j, i]

        def pair_direct_candidates(move_idx, other_idx):
            wi = w_arr[move_idx]
            hi = h_arr[move_idx]
            cur_x, cur_y = place_np[move_idx]
            orig_x, orig_y = original_np[move_idx]
            lj = left[other_idx]
            rj = right[other_idx]
            bj = bottom[other_idx]
            tj = top[other_idx]
            candidates = [(lj - wi / 2.0 - gap, cur_y), (rj + wi / 2.0 + gap, cur_y), (cur_x, bj - hi / 2.0 - gap), (cur_x, tj + hi / 2.0 + gap), (lj - wi / 2.0 - gap, orig_y), (rj + wi / 2.0 + gap, orig_y), (orig_x, bj - hi / 2.0 - gap), (orig_x, tj + hi / 2.0 + gap)]
            uniq = []
            seen = set()
            for cx, cy in candidates:
                cx, cy = clamp_center(move_idx, cx, cy)
                key = (round(cx, 6), round(cy, 6))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((cx, cy))
            uniq.sort(key=lambda p: ((p[0] - cur_x) ** 2 + (p[1] - cur_y) ** 2, (p[0] - orig_x) ** 2 + (p[1] - orig_y) ** 2))
            return uniq

        def strong_global_candidates(i):
            """
            Strong global empty-slot candidates.

            Candidate source:
                1. hard macro boundary x/y coordinates
                2. dense global grid
            """
            wi = w_arr[i]
            hi = h_arr[i]
            orig_x, orig_y = original_np[i]
            cur_x, cur_y = place_np[i]
            x_min = wi / 2.0 + gap
            x_max = canvas_w - wi / 2.0 - gap
            y_min = hi / 2.0 + gap
            y_max = canvas_h - hi / 2.0 - gap
            if x_max < x_min or y_max < y_min:
                return []
            candidates = []
            x_extra = [orig_x, cur_x, x_min, x_max]
            y_extra = [orig_y, cur_y, y_min, y_max]
            x_extra.extend((left[:num_hard] - wi / 2.0 - gap).tolist())
            x_extra.extend((right[:num_hard] + wi / 2.0 + gap).tolist())
            y_extra.extend((bottom[:num_hard] - hi / 2.0 - gap).tolist())
            y_extra.extend((top[:num_hard] + hi / 2.0 + gap).tolist())
            x_extra = [min(max(float(x), x_min), x_max) for x in x_extra]
            y_extra = [min(max(float(y), y_min), y_max) for y in y_extra]
            x_extra = sorted(set((round(x, 6) for x in x_extra)), key=lambda x: abs(x - orig_x))[:edge_top_k]
            y_extra = sorted(set((round(y, 6) for y in y_extra)), key=lambda y: abs(y - orig_y))[:edge_top_k]
            for x in x_extra:
                candidates.append((float(x), float(orig_y)))
                candidates.append((float(x), float(cur_y)))
            for y in y_extra:
                candidates.append((float(orig_x), float(y)))
                candidates.append((float(cur_x), float(y)))
            for x in x_extra:
                for y in y_extra:
                    candidates.append((float(x), float(y)))
            if grid_div is not None and grid_div > 0:
                x_grid = np.linspace(x_min, x_max, int(grid_div))
                y_grid = np.linspace(y_min, y_max, int(grid_div))
                gx, gy = np.meshgrid(x_grid, y_grid)
                flat_x = gx.ravel()
                flat_y = gy.ravel()
                dist2 = (flat_x - orig_x) ** 2 + (flat_y - orig_y) ** 2
                order = np.argsort(dist2)
                if len(order) > top_k_grid:
                    order = order[:top_k_grid]
                for idx in order:
                    candidates.append((float(flat_x[idx]), float(flat_y[idx])))
            uniq = []
            seen = set()
            for cx, cy in candidates:
                cx, cy = clamp_center(i, cx, cy)
                key = (round(cx, 6), round(cy, 6))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((cx, cy))
            uniq.sort(key=lambda p: (p[0] - orig_x) ** 2 + (p[1] - orig_y) ** 2)
            return uniq

        def try_rescue_pair(i, j):
            move_list = move_order_for_pair(i, j)
            if not move_list:
                if verbose:
                    print(f'[FINAL RESCUE][WARN] fixed-fixed pair cannot be repaired: ({i}, {j})', flush=True)
                return False
            for move_idx in move_list:
                other_idx = j if move_idx == i else i
                old_x, old_y = place_np[move_idx]
                for cx, cy in pair_direct_candidates(move_idx, other_idx):
                    if is_legal_hard_position(move_idx, cx, cy):
                        place_np[move_idx, 0] = cx
                        place_np[move_idx, 1] = cy
                        update_rects()
                        if verbose:
                            print(f'[FINAL RESCUE] pair move macro {move_idx}: ({old_x:.4f}, {old_y:.4f}) -> ({cx:.4f}, {cy:.4f}), pair=({i},{j})', flush=True)
                        return True
                for cx, cy in strong_global_candidates(move_idx):
                    if is_legal_hard_position(move_idx, cx, cy):
                        place_np[move_idx, 0] = cx
                        place_np[move_idx, 1] = cy
                        update_rects()
                        if verbose:
                            disp = float(np.sqrt((cx - original_np[move_idx, 0]) ** 2 + (cy - original_np[move_idx, 1]) ** 2))
                            print(f'[FINAL RESCUE] global move macro {move_idx}: ({old_x:.4f}, {old_y:.4f}) -> ({cx:.4f}, {cy:.4f}), disp_from_dp={disp:.4f}, pair=({i},{j})', flush=True)
                        return True
            return False
        restore_fixed()
        update_rects()
        pairs = self.get_official_overlap_pairs_exact(to_output(place_np), benchmark)
        if verbose:
            print(f'[FINAL RESCUE] initial remaining pairs = {len(pairs)}: {pairs[:20]}', flush=True)
        total_moves = 0
        for i, j in pairs:
            ok = try_rescue_pair(i, j)
            if ok:
                total_moves += 1
                restore_fixed()
                update_rects()
        restore_fixed()
        update_rects()
        final_tensor = to_output(place_np)
        final_valid, final_violations = validate_placement(final_tensor, benchmark)
        final_pairs = self.get_official_overlap_pairs_exact(final_tensor, benchmark)
        if verbose:
            print(f'[FINAL RESCUE] final valid={final_valid}, remaining_pairs={len(final_pairs)}, total_moves={total_moves}, runtime={time.time() - t0:.3f}s', flush=True)
            if final_violations:
                print(f'[FINAL RESCUE] final violations: {final_violations}', flush=True)
        return final_tensor

    def clamp_movable_macros_inside_canvas_strict(self, placement, benchmark, eps=0.0001, verbose=True):
        """
        Strictly clamp all non-fixed macros inside canvas.

        Official validator uses strict checks:
            x_min < 0
            x_max > canvas_width
            y_min < 0
            y_max > canvas_height

        Therefore, we keep a small eps margin away from the boundary.

        Fixed macros are never moved.
        """
        import numpy as np
        import torch
        input_is_torch = isinstance(placement, torch.Tensor)
        if input_is_torch:
            device = placement.device
            dtype = placement.dtype
            place_np = placement.detach().cpu().numpy().copy()
        else:
            device = None
            dtype = None
            place_np = np.asarray(placement, dtype=np.float64).copy()
        macro_sizes = benchmark.macro_sizes
        macro_fixed = benchmark.macro_fixed
        macro_positions = benchmark.macro_positions
        if isinstance(macro_sizes, torch.Tensor):
            macro_sizes_np = macro_sizes.detach().cpu().numpy()
        else:
            macro_sizes_np = np.asarray(macro_sizes)
        if isinstance(macro_fixed, torch.Tensor):
            macro_fixed_np = macro_fixed.detach().cpu().numpy().astype(bool)
        else:
            macro_fixed_np = np.asarray(macro_fixed).astype(bool)
        if isinstance(macro_positions, torch.Tensor):
            macro_positions_np = macro_positions.detach().cpu().numpy()
        else:
            macro_positions_np = np.asarray(macro_positions)
        place_np = np.asarray(place_np, dtype=np.float64)
        macro_sizes_np = np.asarray(macro_sizes_np, dtype=np.float64)
        macro_positions_np = np.asarray(macro_positions_np, dtype=np.float64)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        moved = 0
        fixed_outside = []
        for i in range(place_np.shape[0]):
            w = macro_sizes_np[i, 0]
            h = macro_sizes_np[i, 1]
            x_low = w / 2.0 + eps
            x_high = canvas_w - w / 2.0 - eps
            y_low = h / 2.0 + eps
            y_high = canvas_h - h / 2.0 - eps
            if x_low > x_high:
                x_low = x_high = canvas_w / 2.0
            if y_low > y_high:
                y_low = y_high = canvas_h / 2.0
            old_x, old_y = (place_np[i, 0], place_np[i, 1])
            if macro_fixed_np[i]:
                fx, fy = (macro_positions_np[i, 0], macro_positions_np[i, 1])
                lx = fx - w / 2.0
                ux = fx + w / 2.0
                ly = fy - h / 2.0
                uy = fy + h / 2.0
                if lx < 0.0 or ux > canvas_w or ly < 0.0 or (uy > canvas_h):
                    fixed_outside.append(i)
                place_np[i] = macro_positions_np[i]
                continue
            place_np[i, 0] = min(max(float(old_x), x_low), x_high)
            place_np[i, 1] = min(max(float(old_y), y_low), y_high)
            if abs(place_np[i, 0] - old_x) > 1e-12 or abs(place_np[i, 1] - old_y) > 1e-12:
                moved += 1
        if verbose:
            print(f'[BOUND CLAMP] moved movable macros inside canvas: {moved}', flush=True)
            if fixed_outside:
                print(f'[BOUND CLAMP][WARN] fixed macros outside canvas and cannot be moved: {fixed_outside[:50]}', flush=True)
        if input_is_torch:
            return torch.tensor(place_np, dtype=dtype, device=device)
        return place_np

    def get_official_overlap_pairs_exact(self, placement, benchmark):
        """
        Sweep-line version of official hard-macro overlap pair finder.

        Same rule as official validate_placement():
            only check hard macros [0, num_hard_macros)
            overlap if:
                not (lx_i >= ux_j or ux_i <= lx_j or ly_i >= uy_j or uy_i <= ly_j)

        This avoids building a num_hard x num_hard matrix.
        """
        import numpy as np
        import torch
        if isinstance(placement, torch.Tensor):
            placement_np = placement.detach().cpu().numpy()
        else:
            placement_np = np.asarray(placement)
        macro_sizes = benchmark.macro_sizes
        if isinstance(macro_sizes, torch.Tensor):
            macro_sizes_np = macro_sizes.detach().cpu().numpy()
        else:
            macro_sizes_np = np.asarray(macro_sizes)
        num_hard = int(benchmark.num_hard_macros)
        x = placement_np[:num_hard, 0]
        y = placement_np[:num_hard, 1]
        w = macro_sizes_np[:num_hard, 0]
        h = macro_sizes_np[:num_hard, 1]
        lx = x - w / 2.0
        ux = x + w / 2.0
        ly = y - h / 2.0
        uy = y + h / 2.0
        order = np.argsort(lx)
        active = []
        pairs = []
        for idx in order:
            i = int(idx)
            cur_lx = lx[i]
            active = [j for j in active if ux[j] > cur_lx]
            for j in active:
                a = min(i, j)
                b = max(i, j)
                if not (ly[a] >= uy[b] or uy[a] <= ly[b]):
                    pairs.append((a, b))
            active.append(i)
        if len(pairs) > 1:
            pairs = sorted(set(pairs))
        return pairs

    def count_official_overlap_messages(self, violations):
        """
        Count actual overlaps from official validate_placement() messages.

        Example:
            5 explicit messages + "... and 21 more overlaps" = 26 overlaps.
        """
        count = 0
        for v in violations:
            s = str(v)
            if re.search('Macros\\s+\\d+\\s+and\\s+\\d+\\s+overlap', s):
                count += 1
                continue
            m = re.search('\\.\\.\\.\\s+and\\s+(\\d+)\\s+more\\s+overlaps', s)
            if m:
                count += int(m.group(1))
                continue
            count += 1
        return count

    def legalize_identified_hard_overlaps(self, placement, benchmark, overlap_pairs=None, max_rounds=10, grid_div=60, top_k_grid=400, top_k_edges=40, gap=0.001, verbose=False):
        """
        Fast legalizer for official hard-macro overlaps.

        Rules:
            1. Fixed macros never move.
            2. If fixed overlaps movable, only movable can move.
            3. If movable overlaps movable, move high-overlap-degree macro first.
            4. Candidate must be inside canvas and not overlap any hard macro.
            5. Soft macro overlap is ignored, same as official validator.

        Speed improvements:
            1. Internal overlap finder uses sweep-line, not num_hard x num_hard matrix.
            2. Each round computes current overlap pairs once.
            3. The inner loop does not repeatedly call current_official_pairs_fast().
            4. Candidate count is capped.
        """
        import numpy as np
        import torch
        import time
        t0 = time.time()
        input_is_torch = isinstance(placement, torch.Tensor)
        if input_is_torch:
            device = placement.device
            dtype = placement.dtype
            place_np = placement.detach().cpu().numpy().copy()
        else:
            device = None
            dtype = None
            place_np = np.asarray(placement, dtype=np.float64).copy()
        macro_sizes = benchmark.macro_sizes
        macro_fixed = benchmark.macro_fixed
        macro_positions = benchmark.macro_positions
        if isinstance(macro_sizes, torch.Tensor):
            macro_sizes_np = macro_sizes.detach().cpu().numpy()
        else:
            macro_sizes_np = np.asarray(macro_sizes)
        if isinstance(macro_fixed, torch.Tensor):
            macro_fixed_np = macro_fixed.detach().cpu().numpy().astype(bool)
        else:
            macro_fixed_np = np.asarray(macro_fixed).astype(bool)
        if isinstance(macro_positions, torch.Tensor):
            macro_positions_np = macro_positions.detach().cpu().numpy()
        else:
            macro_positions_np = np.asarray(macro_positions)
        place_np = np.asarray(place_np, dtype=np.float64)
        original_np = place_np.copy()
        macro_sizes_np = np.asarray(macro_sizes_np, dtype=np.float64)
        macro_positions_np = np.asarray(macro_positions_np, dtype=np.float64)
        num_macros = int(benchmark.num_macros)
        num_hard = int(benchmark.num_hard_macros)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        w_arr = macro_sizes_np[:, 0]
        h_arr = macro_sizes_np[:, 1]
        top_k_edges = int(top_k_edges)
        top_k_grid = int(top_k_grid)
        grid_div = int(grid_div)
        max_rounds = int(max_rounds)
        left = right = bottom = top = None

        def to_output(arr):
            if input_is_torch:
                return torch.tensor(arr, dtype=dtype, device=device)
            return arr

        def update_rects():
            nonlocal left, right, bottom, top
            left = place_np[:, 0] - w_arr / 2.0
            right = place_np[:, 0] + w_arr / 2.0
            bottom = place_np[:, 1] - h_arr / 2.0
            top = place_np[:, 1] + h_arr / 2.0

        def restore_fixed():
            if macro_fixed_np.any():
                place_np[macro_fixed_np] = macro_positions_np[macro_fixed_np]
            update_rects()

        def clamp_center(i, cx, cy):
            wi = w_arr[i]
            hi = h_arr[i]
            cx = min(max(float(cx), wi / 2.0), canvas_w - wi / 2.0)
            cy = min(max(float(cy), hi / 2.0), canvas_h - hi / 2.0)
            return (cx, cy)

        def clamp_all_movable():
            for i in range(num_macros):
                if macro_fixed_np[i]:
                    continue
                place_np[i, 0], place_np[i, 1] = clamp_center(i, place_np[i, 0], place_np[i, 1])
            restore_fixed()

        def current_official_pairs_fast():
            """
            Sweep-line official-compatible hard overlap finder.
            Avoids num_hard x num_hard matrix.
            """
            lx = left[:num_hard]
            ux = right[:num_hard]
            ly = bottom[:num_hard]
            uy = top[:num_hard]
            order = np.argsort(lx)
            active = []
            pairs = []
            for idx in order:
                i = int(idx)
                cur_lx = lx[i]
                active = [j for j in active if ux[j] > cur_lx]
                for j in active:
                    a = min(i, j)
                    b = max(i, j)
                    if not (ly[a] >= uy[b] or uy[a] <= ly[b]):
                        pairs.append((a, b))
                active.append(i)
            if len(pairs) > 1:
                pairs = sorted(set(pairs))
            return pairs

        def is_legal_hard_position(i, cx, cy):
            """
            Candidate legality:
                inside canvas
                no overlap with any other hard macro
            """
            wi = w_arr[i]
            hi = h_arr[i]
            lx = cx - wi / 2.0
            ux = cx + wi / 2.0
            ly = cy - hi / 2.0
            uy = cy + hi / 2.0
            if lx < 0.0 or ly < 0.0 or ux > canvas_w or (uy > canvas_h):
                return False
            separated = (lx >= right[:num_hard]) | (ux <= left[:num_hard]) | (ly >= top[:num_hard]) | (uy <= bottom[:num_hard])
            overlap = ~separated
            overlap[i] = False
            return not bool(np.any(overlap))

        def build_degree(pairs):
            degree = {}
            for i, j in pairs:
                degree[i] = degree.get(i, 0) + 1
                degree[j] = degree.get(j, 0) + 1
            return degree

        def pair_direct_candidates(move_idx, other_idx):
            """
            Direct candidates around the other macro.
            """
            wi = w_arr[move_idx]
            hi = h_arr[move_idx]
            cur_x, cur_y = place_np[move_idx]
            orig_x, orig_y = original_np[move_idx]
            lj = left[other_idx]
            rj = right[other_idx]
            bj = bottom[other_idx]
            tj = top[other_idx]
            candidates = [(lj - wi / 2.0 - gap, cur_y), (rj + wi / 2.0 + gap, cur_y), (cur_x, bj - hi / 2.0 - gap), (cur_x, tj + hi / 2.0 + gap), (lj - wi / 2.0 - gap, orig_y), (rj + wi / 2.0 + gap, orig_y), (orig_x, bj - hi / 2.0 - gap), (orig_x, tj + hi / 2.0 + gap)]
            uniq = []
            seen = set()
            for cx, cy in candidates:
                cx, cy = clamp_center(move_idx, cx, cy)
                key = (round(cx, 6), round(cy, 6))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((cx, cy))
            uniq.sort(key=lambda p: ((p[0] - cur_x) ** 2 + (p[1] - cur_y) ** 2, (p[0] - orig_x) ** 2 + (p[1] - orig_y) ** 2))
            return uniq

        def global_empty_slot_candidates(i):
            """
            Generate capped global empty-slot candidates.
            """
            wi = w_arr[i]
            hi = h_arr[i]
            orig_x, orig_y = original_np[i]
            cur_x, cur_y = place_np[i]
            x_min = wi / 2.0
            x_max = canvas_w - wi / 2.0
            y_min = hi / 2.0
            y_max = canvas_h - hi / 2.0
            if x_max < x_min or y_max < y_min:
                return []
            candidates = []
            x_extra = [orig_x, cur_x, x_min, x_max]
            y_extra = [orig_y, cur_y, y_min, y_max]
            x_extra.extend((left[:num_hard] - wi / 2.0 - gap).tolist())
            x_extra.extend((right[:num_hard] + wi / 2.0 + gap).tolist())
            y_extra.extend((bottom[:num_hard] - hi / 2.0 - gap).tolist())
            y_extra.extend((top[:num_hard] + hi / 2.0 + gap).tolist())
            x_extra = [min(max(float(x), x_min), x_max) for x in x_extra]
            y_extra = [min(max(float(y), y_min), y_max) for y in y_extra]
            x_extra = sorted(set((round(x, 6) for x in x_extra)), key=lambda x: abs(x - orig_x))[:top_k_edges]
            y_extra = sorted(set((round(y, 6) for y in y_extra)), key=lambda y: abs(y - orig_y))[:top_k_edges]
            for x in x_extra:
                candidates.append((float(x), float(orig_y)))
                candidates.append((float(x), float(cur_y)))
            for y in y_extra:
                candidates.append((float(orig_x), float(y)))
                candidates.append((float(cur_x), float(y)))
            for x in x_extra:
                for y in y_extra:
                    candidates.append((float(x), float(y)))
            if grid_div is not None and grid_div > 0:
                x_grid = np.linspace(x_min, x_max, int(grid_div))
                y_grid = np.linspace(y_min, y_max, int(grid_div))
                gx, gy = np.meshgrid(x_grid, y_grid)
                flat_x = gx.ravel()
                flat_y = gy.ravel()
                dist2 = (flat_x - orig_x) ** 2 + (flat_y - orig_y) ** 2
                order = np.argsort(dist2)[:top_k_grid]
                for idx in order:
                    candidates.append((float(flat_x[idx]), float(flat_y[idx])))
            uniq = []
            seen = set()
            for cx, cy in candidates:
                cx, cy = clamp_center(i, cx, cy)
                key = (round(cx, 6), round(cy, 6))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((cx, cy))
            uniq.sort(key=lambda p: (p[0] - orig_x) ** 2 + (p[1] - orig_y) ** 2)
            return uniq

        def try_move_macro(move_idx, neighbor_list):
            old_x, old_y = place_np[move_idx]
            for other_idx in neighbor_list:
                for cx, cy in pair_direct_candidates(move_idx, other_idx):
                    if is_legal_hard_position(move_idx, cx, cy):
                        place_np[move_idx, 0] = cx
                        place_np[move_idx, 1] = cy
                        update_rects()
                        if verbose:
                            disp = float(np.sqrt((cx - original_np[move_idx, 0]) ** 2 + (cy - original_np[move_idx, 1]) ** 2))
                            print(f'[LEGALIZE-FAST] pair move macro {move_idx}: ({old_x:.4f}, {old_y:.4f}) -> ({cx:.4f}, {cy:.4f}), disp={disp:.4f}', flush=True)
                        return True
            for cx, cy in global_empty_slot_candidates(move_idx):
                if is_legal_hard_position(move_idx, cx, cy):
                    place_np[move_idx, 0] = cx
                    place_np[move_idx, 1] = cy
                    update_rects()
                    if verbose:
                        disp = float(np.sqrt((cx - original_np[move_idx, 0]) ** 2 + (cy - original_np[move_idx, 1]) ** 2))
                        print(f'[LEGALIZE-FAST] global move macro {move_idx}: ({old_x:.4f}, {old_y:.4f}) -> ({cx:.4f}, {cy:.4f}), disp={disp:.4f}', flush=True)
                    return True
            return False
        clamp_all_movable()
        restore_fixed()
        update_rects()
        total_moves = 0
        fixed_fixed_pairs = set()
        for round_id in range(max_rounds):
            current_pairs = current_official_pairs_fast()
            if verbose:
                print(f'[LEGALIZE-FAST] round={round_id}, official overlaps={len(current_pairs)}', flush=True)
            if not current_pairs:
                break
            degree = build_degree(current_pairs)
            neighbors = {}
            for i, j in current_pairs:
                neighbors.setdefault(i, []).append(j)
                neighbors.setdefault(j, []).append(i)
            movable_candidates = []
            for k, deg in degree.items():
                if k >= num_hard:
                    continue
                if macro_fixed_np[k]:
                    continue
                area = w_arr[k] * h_arr[k]
                disp2 = float(np.sum((place_np[k] - original_np[k]) ** 2))
                movable_candidates.append((-deg, area, disp2, k))
            movable_candidates.sort()
            if not movable_candidates:
                for i, j in current_pairs:
                    if macro_fixed_np[i] and macro_fixed_np[j]:
                        fixed_fixed_pairs.add((i, j))
                if verbose:
                    print('[LEGALIZE-FAST] no movable macro in overlap pairs', flush=True)
                break
            moved_this_round = 0
            max_moves_per_round = len(movable_candidates)
            for _, _, _, move_idx in movable_candidates[:max_moves_per_round]:
                neighbor_list = neighbors.get(move_idx, [])
                if not neighbor_list:
                    continue
                ok = try_move_macro(move_idx, neighbor_list)
                if ok:
                    total_moves += 1
                    moved_this_round += 1
                    restore_fixed()
                    update_rects()
            if moved_this_round == 0:
                if verbose:
                    print('[LEGALIZE-FAST] no macro can be moved to legal slot in this round', flush=True)
                break
        restore_fixed()
        update_rects()
        final_tensor = to_output(place_np)
        final_valid, final_violations = validate_placement(final_tensor, benchmark)
        final_pairs = current_official_pairs_fast()
        if verbose:
            print(f'[LEGALIZE-FAST] final valid={final_valid}, final official overlaps={len(final_pairs)}, total_moves={total_moves}', flush=True)
            if fixed_fixed_pairs:
                print(f'[LEGALIZE-FAST] fixed-fixed pairs: {sorted(fixed_fixed_pairs)[:20]}', flush=True)
            if final_violations:
                print(f'[LEGALIZE-FAST] final violations: {final_violations[:10]}', flush=True)
            print(f'[LEGALIZE-FAST] runtime={time.time() - t0:.3f}s', flush=True)
        return final_tensor

    def patch_case_weight_zero_to_one(self, weight_path, backup=True):
        """
        Only patch the current case's .weight file.

        Convert:
            net_5672 0
        to:
            net_5672 1
        """
        weight_path = Path(weight_path)
        if not weight_path.exists():
            print(f'[PATCH WEIGHT] Skip, file not found: {weight_path}')
            return 0
        text = weight_path.read_text(errors='ignore')
        changed_count = 0
        new_lines = []
        for line in text.splitlines(keepends=True):
            raw = line.rstrip('\n')
            newline = '\n' if line.endswith('\n') else ''
            parts = raw.split()
            if len(parts) == 2 and parts[1] == '0':
                new_lines.append(f'{parts[0]} 1{newline}')
                changed_count += 1
            else:
                new_lines.append(line)
        if changed_count == 0:
            print(f'[PATCH WEIGHT] No zero weights in: {weight_path}')
            return 0
        if backup:
            backup_path = weight_path.with_suffix(weight_path.suffix + '.bak_zero')
            if not backup_path.exists():
                backup_path.write_text(text)
                print(f'[PATCH WEIGHT] Backup saved: {backup_path}')
        weight_path.write_text(''.join(new_lines))
        print(f'[PATCH WEIGHT] Patched {changed_count} zero weights in: {weight_path}')
        return changed_count

    def patch_pbtxt_scientific_float_inplace(self, netlist_path, zero_threshold='1e-12', backup=True):
        """
        In-place patch netlist.pb.txt.

        Convert protobuf float fields like:
            f: 5.68434e-16

        into:
            f: 0.0

        or normal decimal form.
        """
        netlist_path = Path(netlist_path)
        if not netlist_path.exists():
            raise FileNotFoundError(f'netlist.pb.txt not found: {netlist_path}')
        text = netlist_path.read_text(errors='ignore')
        zero_th = Decimal(str(zero_threshold))
        sci_float_re = re.compile('(?P<prefix>\\bf:\\s*)(?P<num>[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)[eE][-+]?\\d+)(?=\\s|$)')
        changed_count = 0

        def repl(m):
            nonlocal changed_count
            prefix = m.group('prefix')
            num_str = m.group('num')
            try:
                value = Decimal(num_str)
            except InvalidOperation:
                return m.group(0)
            changed_count += 1
            if abs(value) < zero_th:
                return prefix + '0.0'
            out = format(value, 'f')
            if '.' in out:
                out = out.rstrip('0').rstrip('.')
            if out in ('', '+', '-', '0', '+0', '-0'):
                out = '0.0'
            return prefix + out
        new_text = sci_float_re.sub(repl, text)
        if changed_count == 0:
            print(f'[PATCH PBTXT] No scientific float field found: {netlist_path}')
            return 0
        if backup:
            backup_path = netlist_path.with_suffix(netlist_path.suffix + '.bak_sci')
            if not backup_path.exists():
                backup_path.write_text(text)
                print(f'[PATCH PBTXT] Backup saved: {backup_path}')
        netlist_path.write_text(new_text)
        print(f'[PATCH PBTXT] Patched {changed_count} scientific float fields in-place')
        print(f'[PATCH PBTXT] File: {netlist_path}')
        return changed_count

    def _bool_env(self, name, default=False):
        value = os.environ.get(name, None)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

    def _case_name_from_text(self, text):
        if text is None:
            return None
        m = re.search('\\b(ibm\\d+)\\b', str(text), flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return None

    def _case_name_from_argv(self):
        """
        Support commands like:
            uv run evaluate submissions/examples/my_placer.py -b ibm01
            uv run evaluate submissions/examples/my_placer.py --benchmark ibm01
        """
        argv = list(sys.argv)
        for flag in ('-b', '--benchmark'):
            if flag in argv:
                k = argv.index(flag)
                if k + 1 < len(argv):
                    case = self._case_name_from_text(argv[k + 1])
                    if case:
                        return case
        for arg in argv:
            case = self._case_name_from_text(arg)
            if case:
                return case
        return None

    def _infer_case_name_from_benchmark(self, benchmark):
        """
        Try several common attributes. If the Benchmark object does not expose a
        name, fall back to parsing sys.argv, because evaluate passes '-b ibmXX'.
        """
        for attr in ('case_name', 'benchmark_name', 'design_name', 'name', 'circuit_name', 'bench_name', 'benchmark_dir', 'root_dir', 'data_dir'):
            if hasattr(benchmark, attr):
                case = self._case_name_from_text(getattr(benchmark, attr))
                if case:
                    return case
        try:
            for _, value in vars(benchmark).items():
                if isinstance(value, (str, Path)):
                    case = self._case_name_from_text(value)
                    if case:
                        return case
        except Exception:
            pass
        return self._case_name_from_argv()

    def _resolve_json_file(self, benchmark=None, json_file=None):
        if json_file is not None:
            json_path = Path(json_file)
            if json_path.exists():
                return json_path
            raise FileNotFoundError(f'JSON file not found: {json_path}')
        case_name = None
        if benchmark is not None:
            case_name = self._infer_case_name_from_benchmark(benchmark)
        if case_name is None:
            case_name = self._case_name_from_argv()
        candidate_dirs = [self.run_json_dir, Path.cwd() / 'run_json', Path.cwd() / 'macro-place-challenge-2026' / 'run_json', Path('/Macro_challenge_2026/macro-place-challenge-2026/run_json')]
        if case_name is not None:
            for d in candidate_dirs:
                p = d / f'{case_name}.json'
                if p.exists():
                    return p
            raise FileNotFoundError(f'Cannot find {case_name}.json in candidate run_json dirs: {[str(d) for d in candidate_dirs]}')
        json_files = []
        for d in candidate_dirs:
            if d.exists():
                json_files.extend(sorted(d.glob('ibm*.json')))
        json_files = list(dict.fromkeys(json_files))
        if len(json_files) == 1:
            return json_files[0]
        raise RuntimeError("Cannot infer benchmark case name. Please run evaluate with '-b ibmXX' or set MPC_RUN_JSON_DIR to the directory containing ibmXX.json.")

    def _restore_fixed_macros(self, placement, benchmark):
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return placement

    def greedy_row_fallback(self, benchmark, gap=0.001):
        """
        A safe fallback copied from the demo idea. It only places movable hard macros.
        This keeps the evaluator from crashing if DREAMPlace fails.
        """
        placement = benchmark.macro_positions.clone()
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        sizes = benchmark.macro_sizes
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        movable_indices.sort(key=lambda i: -sizes[i, 1].item())
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0
        for idx in movable_indices:
            w = float(sizes[idx, 0].item())
            h = float(sizes[idx, 1].item())
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0
            if cursor_y + h > canvas_h:
                cx = min(max(w / 2.0 + gap, w / 2.0), canvas_w - w / 2.0 - gap)
                cy = min(max(h / 2.0 + gap, h / 2.0), canvas_h - h / 2.0 - gap)
                placement[idx, 0] = cx
                placement[idx, 1] = cy
                continue
            placement[idx, 0] = cursor_x + w / 2.0
            placement[idx, 1] = cursor_y + h / 2.0
            cursor_x += w + gap
            row_height = max(row_height, h)
        return self._restore_fixed_macros(placement, benchmark)

    def run_placement(self, json_file=None, benchmark=None, verbose=False, auto_cluster=False):
        """
        Single-case placement entry.

        Input:
            json_file : optional path to ibmXX.json
            benchmark : Benchmark object passed by evaluate

        Output:
            placement : torch.Tensor, shape [num_macros, 2]

        This function is different from the old experiment script version:
            old version returned a result dict;
            this version returns the placement tensor required by evaluate.
        """
        json_path = self._resolve_json_file(benchmark=benchmark, json_file=json_file)
        params = Params.Params()
        params.load(str(json_path))
        params.regioning = False
        loaded_benchmark, raw_plc = load_benchmark_from_dir(params.benchmark_dir)
        if benchmark is None:
            benchmark = loaded_benchmark
        netlist_file = os.path.join(params.benchmark_dir, 'netlist.pb.txt')
        plc_file = os.path.join(params.benchmark_dir, 'initial.plc')
        if verbose:
            print(f'[INFO] json_file   : {json_path}', flush=True)
            print(f'[INFO] netlist_file: {netlist_file}', flush=True)
            print(f'[INFO] plc_file    : {plc_file}', flush=True)
            print(f'[INFO] case_name   : {params.case_name}', flush=True)
        weight_path = os.path.join(params.benchmark_dir, f'{params.case_name}.weight')
        if auto_cluster and (not os.path.exists(weight_path)):
            if verbose:
                print(f'[INFO] Weight file missing, run one default cluster: {weight_path}', flush=True)
            self.run_cluster_for_case(bench_dir=Path(params.benchmark_dir), bench_name=params.case_name, cluster_params={'max_num_inst': int(os.environ.get('MPC_CLUSTER_MAX_INST', '1000')), 'min_num_inst': int(os.environ.get('MPC_CLUSTER_MIN_INST', '50')), 'seed': int(os.environ.get('MPC_CLUSTER_SEED', '1'))})
        placedb = build_direct_placedb_from_plc(raw_plc=raw_plc, params=params, design_name=params.case_name, debug=verbose)
        if os.path.exists(weight_path):
            placedb = apply_net_weights_from_file(placedb=placedb, weight_path=weight_path, default_weight=1.0, min_weight=1.0, verbose=verbose)
        elif verbose:
            print(f'[WARN] Weight file not found, skip net weight loading: {weight_path}', flush=True)
        if verbose:
            debug_direct_placedb_hpwl(placedb)
            print('========== PARAM FLAGS ==========', flush=True)
            for name in ['global_place_flag', 'macro_place_flag', 'legalize_flag', 'detailed_place_flag', 'enable_fillers', 'routability_opt_flag', 'timing_opt_flag', 'random_center_init_flag', 'plot_flag']:
                if hasattr(params, name):
                    print(f'{name:<25} = {getattr(params, name)}', flush=True)
            print('=================================', flush=True)
        self.dreamplace_interface(params, placedb)
        placement = extract_macro_placement_from_placedb(placedb=placedb, benchmark=benchmark, raw_plc=raw_plc, params=params)
        placement = self._restore_fixed_macros(placement, benchmark)
        is_valid, violations = validate_placement(placement, benchmark)
        official_pairs = self.get_official_overlap_pairs_exact(placement, benchmark)
        official_pairs_after = official_pairs
        if verbose:
            print(f'[INFO] Before legalize: valid={is_valid}, official_overlap_pairs={len(official_pairs)}, official_overlap_count_from_messages={self.count_official_overlap_messages(violations)}, violation_messages={len(violations)}', flush=True)
            if violations:
                print(f'[INFO] Before legalize violations: {violations[:10]}', flush=True)
        if not is_valid and official_pairs:
            if verbose:
                print('[INFO] Start first-pass legalization...', flush=True)
            placement = self.legalize_identified_hard_overlaps(placement=placement, benchmark=benchmark, overlap_pairs=official_pairs, max_rounds=30, grid_div=120, top_k_grid=8000, top_k_edges=160, gap=0.001, verbose=verbose)
            placement = self.clamp_movable_macros_inside_canvas_strict(placement=placement, benchmark=benchmark, eps=0.0001, verbose=verbose)
            placement = self._restore_fixed_macros(placement, benchmark)
            is_valid, violations = validate_placement(placement, benchmark)
            official_pairs_after = self.get_official_overlap_pairs_exact(placement, benchmark)
            if verbose:
                print(f'[INFO] After first-pass legalize: valid={is_valid}, official_overlap_pairs={len(official_pairs_after)}, violations={violations[:10]}', flush=True)
            if not is_valid and official_pairs_after:
                if verbose:
                    print('[INFO] Start second-pass aggressive legalization...', flush=True)
                placement = self.legalize_identified_hard_overlaps(placement=placement, benchmark=benchmark, overlap_pairs=official_pairs_after, max_rounds=100, grid_div=250, top_k_grid=50000, top_k_edges=500, gap=0.001, verbose=verbose)
                placement = self._restore_fixed_macros(placement, benchmark)
                is_valid, violations = validate_placement(placement, benchmark)
                official_pairs_after = self.get_official_overlap_pairs_exact(placement, benchmark)
            if not is_valid and official_pairs_after:
                if verbose:
                    print('[INFO] Start final-rescue legalization...', flush=True)
                placement = self.final_rescue_remaining_overlaps(placement=placement, benchmark=benchmark, grid_div=600, top_k_grid=300000, edge_top_k=1200, gap=0.001, verbose=verbose)
                placement = self._restore_fixed_macros(placement, benchmark)
                is_valid, violations = validate_placement(placement, benchmark)
                official_pairs_after = self.get_official_overlap_pairs_exact(placement, benchmark)
        placement = self.clamp_movable_macros_inside_canvas_strict(placement=placement, benchmark=benchmark, eps=0.0001, verbose=verbose)
        placement = self._restore_fixed_macros(placement, benchmark)
        placement = placement.to(dtype=benchmark.macro_positions.dtype, device=benchmark.macro_positions.device)
        if verbose:
            is_valid, violations = validate_placement(placement, benchmark)
            official_pairs_after = self.get_official_overlap_pairs_exact(placement, benchmark)
            print(f'[INFO] Final placement: valid={is_valid}, official_overlap_pairs={len(official_pairs_after)}, violation_messages={len(violations)}', flush=True)
            try:
                costs = compute_proxy_cost(placement, benchmark, raw_plc)
                proxy_cost = costs['proxy_cost']
                if hasattr(proxy_cost, 'item'):
                    proxy_cost = proxy_cost.item()
                print(f'[INFO] Proxy cost: {float(proxy_cost):.6f}', flush=True)
            except Exception as e:
                print(f'[WARN] compute_proxy_cost failed: {repr(e)}', flush=True)
        return placement

    def build_cluster_param_grid(self):
        """
        Clustering hyperparameter search space.

        Adjust here for full search:
            max_num_inst_list = [1000, 3000, 5000]
            min_num_inst_list = [50, 150, 300]
            seed_list = [1]
        """
        max_num_inst_list = [100, 300, 500, 1000, 1500, 2000, 3000]
        min_num_inst_list = [10, 50]
        seed_list = [1]
        grid = []
        for max_num_inst in max_num_inst_list:
            for min_num_inst in min_num_inst_list:
                for seed in seed_list:
                    if min_num_inst >= max_num_inst:
                        continue
                    grid.append({'max_num_inst': max_num_inst, 'min_num_inst': min_num_inst, 'seed': seed})
        return grid

    def run_cluster_for_case(self, bench_dir, bench_name, cluster_params):
        """
        Run clustering once for one case with given clustering parameters.

        It writes:
            bench_dir / f"{bench_name}.weight"
        """
        bench_dir = Path(bench_dir)
        netlist_path = bench_dir / 'netlist.pb.txt'
        plc_path = bench_dir / 'initial.plc'
        if not netlist_path.exists():
            raise FileNotFoundError(f'netlist.pb.txt not found: {netlist_path}')
        if not plc_path.exists():
            raise FileNotFoundError(f'initial.plc not found: {plc_path}')
        print('=' * 100, flush=True)
        print(f'[RUN CLUSTER] {bench_name}', flush=True)
        print(f'[BENCH DIR  ] {bench_dir}', flush=True)
        print(f'[NETLIST    ] {netlist_path}', flush=True)
        print(f'[PLC        ] {plc_path}', flush=True)
        print(f'[PARAMS     ] {cluster_params}', flush=True)
        print('=' * 100, flush=True)
        self.patch_pbtxt_scientific_float_inplace(netlist_path, zero_threshold='1e-12', backup=True)
        cluster(pt=None, benchmark_dir=str(bench_dir), plc_netlist=str(netlist_path), plc_initial=str(plc_path), nodes=None, nets=None, pl=None, out=str(bench_dir), file_name=bench_name, macro_list=None, macro_area_ratio=10.0, max_num_macro=1, min_num_macro=1, max_num_inst=cluster_params['max_num_inst'], min_num_inst=cluster_params['min_num_inst'], seed=cluster_params['seed'], net_threshold=0, virtual_weight=10, ignore_net_threshold=0, max_net_degree=100, enable_virtual_map_weight=True, enable_timing_like_virtual=True, macro_cell_weight=4, cell_macro_weight=1, macro_macro_weight=2, macro_cell_cell_weight=2)
        weight_path = bench_dir / f'{bench_name}.weight'
        self.patch_case_weight_zero_to_one(weight_path=weight_path, backup=True)
        if not weight_path.exists():
            raise FileNotFoundError(f'weight file not generated: {weight_path}')
        print(f'[CLUSTER DONE] weight file: {weight_path}', flush=True)
        return weight_path

    def result_rank_key(self, result):
        """
        Smaller is better.

        Priority:
          1. valid result first
          2. fewer official overlap pairs
          3. lower proxy cost
        """
        valid = bool(result.get('valid', False))
        num_violations = int(result.get('num_violations', 10 ** 9))
        proxy_cost = result.get('proxy_cost', None)
        if proxy_cost is None:
            proxy_cost = float('inf')
        else:
            proxy_cost = float(proxy_cost)
        return (0 if valid else 1, num_violations, proxy_cost)

    def setup_logging_to_file_only(self, log_dir='/Macro_challenge_2026/macro-place-challenge-2026/logs', log_name='run_all_cases.log'):
        """
        Redirect logging / print / stderr to a single log file.
        No terminal output.
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.flush()
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
        self._log_file_handle = open(log_path, 'w')
        sys.stdout = self._log_file_handle
        sys.stderr = self._log_file_handle
        logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s', stream=sys.stdout, force=True)
        print(f'[LOG] Output redirected to: {log_path}', flush=True)
        return log_path
