# -*- coding: utf-8 -*-

import logging
import numpy as np
import torch

from dreamplace import PlaceDB


def _to_bytes_array(items):
    return np.array(
        [x if isinstance(x, bytes) else str(x).encode("utf-8") for x in items],
        dtype=np.bytes_,
    )


def _flatten_nested_map(nested_map):
    flat = []
    start = [0]

    for row in nested_map:
        if isinstance(row, np.ndarray):
            row = row.tolist()

        for v in row:
            flat.append(int(v))

        start.append(len(flat))

    return (
        np.asarray(flat, dtype=np.int32),
        np.asarray(start, dtype=np.int32),
    )


def _get_name(obj):
    if hasattr(obj, "get_name"):
        return obj.get_name()
    return getattr(obj, "name")


def _get_pos(obj):
    if hasattr(obj, "get_pos"):
        x, y = obj.get_pos()
        return float(x), float(y)

    return float(getattr(obj, "x", 0.0)), float(getattr(obj, "y", 0.0))


def _get_size(obj):
    if hasattr(obj, "get_width"):
        w = obj.get_width()
    else:
        w = getattr(obj, "width", 0.0)

    if hasattr(obj, "get_height"):
        h = obj.get_height()
    else:
        h = getattr(obj, "height", 0.0)

    return float(w), float(h)


def _get_orient(obj):
    if hasattr(obj, "get_orientation"):
        orient = obj.get_orientation()
    elif hasattr(obj, "orientation"):
        orient = obj.orientation
    elif hasattr(obj, "orient"):
        orient = obj.orient
    else:
        orient = "N"

    if isinstance(orient, bytes):
        return orient

    return str(orient).encode("utf-8")


def _normalize_name(name):
    s = str(name).strip().strip("\"'")
    s = s.lstrip("^")
    if ":" in s:
        s = s.split(":")[0]
    return s


def _set_default_params(params, design_name="ibm01"):
    """
    Keep DREAMPlace params compatible with a pre-built PlaceDB.
    """

    params.circuit_training_mode = True

    params.legalize_flag = False
    params.detailed_place_flag = False
    params.enable_fillers = False
    params.random_center_init_flag = True
    params.routability_opt_flag = False
    params.timing_opt_flag = False


    params.design_name = lambda: design_name

    if not hasattr(params, "shift_factor") or params.shift_factor is None:
        params.shift_factor = [0.0, 0.0]

    if not hasattr(params, "scale_factor"):
        params.scale_factor = 1.0

    if params.scale_factor is None:
        params.scale_factor = 1.0

    if not hasattr(params, "max_net_weight"):
        params.max_net_weight = 1.0

    if not hasattr(params, "num_bins_x"):
        params.num_bins_x = 0

    if not hasattr(params, "num_bins_y"):
        params.num_bins_y = 0

    return params


def _scale_placedb_like_dreamplace(db, params):
    """
    Equivalent to the important part of PlaceDB.initialize():
      - set shift_factor
      - set scale_factor
      - scale coordinates, sizes, pins, rows, routing grid, area

    This avoids calling placedb(params).
    """

    params.shift_factor[0] = db.xl
    params.shift_factor[1] = db.yl

    if params.scale_factor == 0.0 or db.site_width != 1.0:
        params.scale_factor = 1.0 / db.site_width

    shift_x, shift_y = params.shift_factor
    scale = params.scale_factor

    if shift_x == 0.0 and shift_y == 0.0 and scale == 1.0:
        return db

    # placement
    db.node_x = (db.node_x - shift_x) * scale
    db.node_y = (db.node_y - shift_y) * scale

    # size
    db.node_size_x = db.node_size_x * scale
    db.node_size_y = db.node_size_y * scale
    db.original_node_size_x = db.original_node_size_x * scale
    db.original_node_size_y = db.original_node_size_y * scale

    # pin offset
    db.pin_offset_x = db.pin_offset_x * scale
    db.pin_offset_y = db.pin_offset_y * scale

    # bbox
    db.xl = (db.xl - shift_x) * scale
    db.yl = (db.yl - shift_y) * scale
    db.xh = (db.xh - shift_x) * scale
    db.yh = (db.yh - shift_y) * scale

    # row
    box_shift = np.array([shift_x, shift_y, shift_x, shift_y], dtype=db.dtype)
    db.rows = (db.rows - box_shift) * scale

    db.row_height *= scale
    db.site_width *= scale

    # routing grid
    db.routing_grid_xl = (db.routing_grid_xl - shift_x) * scale
    db.routing_grid_yl = (db.routing_grid_yl - shift_y) * scale
    db.routing_grid_xh = (db.routing_grid_xh - shift_x) * scale
    db.routing_grid_yh = (db.routing_grid_yh - shift_y) * scale

    # regions
    if db.flat_region_boxes is not None and len(db.flat_region_boxes):
        region_shift = np.array([shift_x, shift_y, shift_x, shift_y], dtype=db.dtype)
        db.flat_region_boxes = (db.flat_region_boxes - region_shift) * scale

    if db.regions is not None:
        for i in range(len(db.regions)):
            if len(db.regions[i]):
                db.regions[i] = (db.regions[i] - box_shift) * scale

    # area
    db.total_space_area *= scale * scale

    return db


def _compute_area_and_macro_info(db, params):
    """
    Equivalent to the important area/macro/filler part of PlaceDB.initialize().
    """

    db.num_terminal_NIs = 0 if db.num_terminal_NIs is None else db.num_terminal_NIs

    db.total_movable_node_area = float(
        np.sum(
            db.node_size_x[: db.num_movable_nodes]
            * db.node_size_y[: db.num_movable_nodes]
        )
    )

    fixed_l = db.num_movable_nodes
    fixed_r = db.num_physical_nodes - db.num_terminal_NIs

    if fixed_r > fixed_l:
        fixed_w = np.maximum(
            np.minimum(db.node_x[fixed_l:fixed_r] + db.node_size_x[fixed_l:fixed_r], db.xh)
            - np.maximum(db.node_x[fixed_l:fixed_r], db.xl),
            0.0,
        )
        fixed_h = np.maximum(
            np.minimum(db.node_y[fixed_l:fixed_r] + db.node_size_y[fixed_l:fixed_r], db.yh)
            - np.maximum(db.node_y[fixed_l:fixed_r], db.yl),
            0.0,
        )
        db.total_fixed_node_area = float(np.sum(fixed_w * fixed_h))
    else:
        db.total_fixed_node_area = 0.0

    # total_space_area: placeable area excluding fixed nodes.
    db.total_space_area = max(float(db.area - db.total_fixed_node_area), 1e-12)

    movable_area = (
        db.node_size_x[: db.num_movable_nodes]
        * db.node_size_y[: db.num_movable_nodes]
    )

    if db.num_movable_nodes > 0:
        mean_movable_area = db.total_movable_node_area / db.num_movable_nodes
    else:
        mean_movable_area = 0.0

    db.movable_macro_mask = np.zeros(db.num_movable_nodes, dtype=bool)

    if db.num_movable_nodes > 0 and mean_movable_area > 0:
        db.movable_macro_mask[:] = (
            (movable_area > mean_movable_area * 10.0)
            & (db.node_size_y[: db.num_movable_nodes] > db.row_height * 2.0)
        )

    db.macro_mask = db.movable_macro_mask.astype(np.uint8)

    db.movable_macro_pins = np.isin(
        db.pin2node_map,
        np.arange(0, db.num_movable_nodes, dtype=np.int32)[db.movable_macro_mask],
    )

    db.total_movable_macro_area = float(movable_area[db.movable_macro_mask].sum())
    db.total_movable_cell_area = float(
        db.total_movable_node_area - db.total_movable_macro_area
    )

    total_cell_space_area = max(
        db.total_space_area - db.total_movable_macro_area,
        1e-12,
    )

    cell_utilization = db.total_movable_cell_area / total_cell_space_area

    if hasattr(params, "target_density"):
        if params.target_density < cell_utilization:
            logging.warning(
                "target_density %g is smaller than cell utilization %g, ignored",
                params.target_density,
                cell_utilization,
            )
            params.target_density = min(cell_utilization, 1.0)
    else:
        params.target_density = min(max(cell_utilization, 0.1), 1.0)

    # Fillers: debug 阶段先关闭。
    if not getattr(params, "enable_fillers", False):
        db.num_filler_nodes = 0
        db.total_filler_node_area = 0.0
        return db

    # 如果后续确实要开 fillers，可以走这里。
    node_size_order = np.argsort(db.node_size_x[: db.num_movable_nodes])
    range_lb = int(db.num_movable_nodes * 0.05)
    range_ub = int(db.num_movable_nodes * 0.95)

    if range_lb >= range_ub:
        filler_size_x = 0.0
    else:
        filler_size_x = float(np.mean(db.node_size_x[node_size_order[range_lb:range_ub]]))

    filler_size_y = float(db.row_height)

    total_filler_area = max(
        total_cell_space_area * params.target_density - db.total_movable_cell_area,
        0.0,
    )

    filler_area = filler_size_x * filler_size_y

    if filler_area <= 0.0:
        db.num_filler_nodes = 0
        db.total_filler_node_area = 0.0
        return db

    db.total_filler_node_area = total_filler_area
    db.num_filler_nodes = int(round(total_filler_area / filler_area))

    if db.num_filler_nodes > 0:
        db.node_size_x = np.concatenate(
            [
                db.node_size_x,
                np.full(db.num_filler_nodes, filler_size_x, dtype=db.dtype),
            ]
        )
        db.node_size_y = np.concatenate(
            [
                db.node_size_y,
                np.full(db.num_filler_nodes, filler_size_y, dtype=db.dtype),
            ]
        )

    return db


def _initialize_bins(db, params, default_bins_x, default_bins_y):
    """
    Avoid calling PlaceDB.initialize_num_bins().
    """

    if getattr(params, "num_bins_x", 0) is None or params.num_bins_x <= 1:
        db.num_bins_x = int(default_bins_x)
        params.num_bins_x = int(default_bins_x)
    else:
        db.num_bins_x = int(params.num_bins_x)

    if getattr(params, "num_bins_y", 0) is None or params.num_bins_y <= 1:
        db.num_bins_y = int(default_bins_y)
        params.num_bins_y = int(default_bins_y)
    else:
        db.num_bins_y = int(params.num_bins_y)

    db.bin_size_x = (db.xh - db.xl) / db.num_bins_x
    db.bin_size_y = (db.yh - db.yl) / db.num_bins_y

    return db


def build_direct_placedb_from_plc(raw_plc, params, design_name="ibm01", debug=True):
    """
    Build a fully initialized DREAMPlace PlaceDB directly from your parsed PLC object.

    This function does NOT call:
      - db.read(params)
      - db.initialize(params)
      - db(params)
      - place_io
      - plc_converter

    After this function:
      dreamplace_interface(params, placedb)
    """

    params = _set_default_params(params, design_name=design_name)

    db = PlaceDB.PlaceDB()
    db.dtype = np.float32
    db.device = torch.device("cuda" if params.gpu else "cpu")

    modules = raw_plc.modules_w_pins

    # ============================================================
    # 1. Canvas / rows
    # ============================================================
    canvas_w, canvas_h = raw_plc.get_canvas_width_height()
    grid_cols = int(raw_plc.grid_col)
    grid_rows = int(raw_plc.grid_row)

    db.xl = 0.0
    db.yl = 0.0
    db.xh = float(canvas_w)
    db.yh = float(canvas_h)

    db.row_height = db.yh / grid_rows
    db.site_width = db.xh / grid_cols

    db.rows = np.array(
        [
            [0.0, i * db.row_height, db.xh, (i + 1) * db.row_height]
            for i in range(grid_rows)
        ],
        dtype=db.dtype,
    )

    # ============================================================
    # 2. Node order: movable first, fixed later.
    # ============================================================
    hard_indices = list(getattr(raw_plc, "hard_macro_indices", []))
    soft_indices = list(getattr(raw_plc, "soft_macro_indices", []))
    port_indices = list(getattr(raw_plc, "port_indices", []))
    pin_indices = list(getattr(raw_plc, "hard_macro_pin_indices", []))

    movable_plc_indices = []
    fixed_plc_indices = []

    # Soft macros / stdcell clusters are movable unless fixed flag is set.
    for idx in soft_indices:
        node = modules[idx]
        fixed = bool(node.get_fix_flag()) if hasattr(node, "get_fix_flag") else False
        if fixed:
            fixed_plc_indices.append(idx)
        else:
            movable_plc_indices.append(idx)

    # Hard macros and ports are fixed terminals in this first version.
    fixed_plc_indices += hard_indices
    fixed_plc_indices += port_indices

    # Dedup while preserving order.
    movable_plc_indices = list(dict.fromkeys(movable_plc_indices))
    fixed_plc_indices = list(dict.fromkeys(fixed_plc_indices))

    physical_plc_indices = movable_plc_indices + fixed_plc_indices

    plc_idx_to_db_id = {}
    name_to_plc_idx = {}

    for i, node in enumerate(modules):
        if hasattr(node, "get_name"):
            name_to_plc_idx[_get_name(node)] = i
            name_to_plc_idx[_normalize_name(_get_name(node))] = i

    db.node_names = []
    db.node_name2id_map = {}
    db.node_x = []
    db.node_y = []
    db.node_orient = []
    db.node_size_x = []
    db.node_size_y = []
    db.original_node_size_x = []
    db.original_node_size_y = []

    for db_id, plc_idx in enumerate(physical_plc_indices):
        node = modules[plc_idx]
        name = _get_name(node)

        plc_idx_to_db_id[plc_idx] = db_id

        db.node_names.append(name)
        db.node_name2id_map[name] = db_id

        x_center, y_center = _get_pos(node)

        if plc_idx in port_indices:
            w, h = 0.0, 0.0
            orient = b"N"
        else:
            w, h = _get_size(node)
            orient = _get_orient(node)

        # PLC uses center coordinate.
        # DREAMPlace uses lower-left coordinate.
        db.node_x.append(x_center - w / 2.0)
        db.node_y.append(y_center - h / 2.0)
        db.node_orient.append(orient)
        db.node_size_x.append(w)
        db.node_size_y.append(h)
        db.original_node_size_x.append(w)
        db.original_node_size_y.append(h)

    # ============================================================
    # 2.1 Save PLC <-> PlaceDB mapping.
    # These mappings are required to extract final macro placement.
    # ============================================================
    db.plc_idx_to_db_id = plc_idx_to_db_id
    db.db_id_to_plc_idx = {v: k for k, v in plc_idx_to_db_id.items()}

    db.movable_plc_indices = movable_plc_indices
    db.fixed_plc_indices = fixed_plc_indices
    db.physical_plc_indices = physical_plc_indices

    db.hard_macro_plc_indices = hard_indices
    db.soft_macro_plc_indices = soft_indices
    db.port_plc_indices = port_indices
    db.hard_macro_pin_plc_indices = pin_indices

    db.node_names = _to_bytes_array(db.node_names)
    db.node_x = np.asarray(db.node_x, dtype=db.dtype)
    db.node_y = np.asarray(db.node_y, dtype=db.dtype)
    db.node_orient = np.asarray(db.node_orient, dtype=np.bytes_)
    db.node_size_x = np.asarray(db.node_size_x, dtype=db.dtype)
    db.node_size_y = np.asarray(db.node_size_y, dtype=db.dtype)
    db.original_node_size_x = np.asarray(db.original_node_size_x, dtype=db.dtype)
    db.original_node_size_y = np.asarray(db.original_node_size_y, dtype=db.dtype)

    db.num_physical_nodes = len(physical_plc_indices)
    db.num_terminals = len(fixed_plc_indices)
    db.num_terminal_NIs = 0
    db.num_non_movable_macros = len(hard_indices)

    # ============================================================
    # 3. Pin / endpoint resolver.
    # ============================================================
    physical_name_to_db_id = {}

    for plc_idx in physical_plc_indices:
        node = modules[plc_idx]
        name = _get_name(node)
        physical_name_to_db_id[name] = plc_idx_to_db_id[plc_idx]
        physical_name_to_db_id[_normalize_name(name)] = plc_idx_to_db_id[plc_idx]

    hard_pin_info = {}

    for pin_idx in pin_indices:
        pin = modules[pin_idx]

        if not hasattr(pin, "get_name"):
            continue

        pin_name = _get_name(pin)
        pin_name_norm = _normalize_name(pin_name)

        if hasattr(pin, "get_macro_name"):
            macro_name = pin.get_macro_name()
        else:
            macro_name = pin_name.split("/")[0]

        macro_name_norm = _normalize_name(macro_name)

        if macro_name not in name_to_plc_idx and macro_name_norm not in name_to_plc_idx:
            continue

        parent_plc_idx = name_to_plc_idx.get(
            macro_name,
            name_to_plc_idx.get(macro_name_norm),
        )

        if parent_plc_idx not in plc_idx_to_db_id:
            continue

        parent_db_id = plc_idx_to_db_id[parent_plc_idx]
        parent_node = modules[parent_plc_idx]
        parent_w, parent_h = _get_size(parent_node)

        # CT-style hard macro pin offset is usually relative to macro center.
        x_off_center = float(getattr(pin, "x_offset", 0.0))
        y_off_center = float(getattr(pin, "y_offset", 0.0))

        # DREAMPlace offset is relative to lower-left.
        offset_x = x_off_center + parent_w / 2.0
        offset_y = y_off_center + parent_h / 2.0

        hard_pin_info[pin_name] = (parent_db_id, offset_x, offset_y)
        hard_pin_info[pin_name_norm] = (parent_db_id, offset_x, offset_y)

    def resolve_endpoint(endpoint):
        """
        Return:
            (parent_db_id, offset_x, offset_y)
        or:
            None
        """

        # Endpoint may be an integer plc index.
        if isinstance(endpoint, int):
            if 0 <= endpoint < len(modules):
                plc_idx = endpoint

                if plc_idx in plc_idx_to_db_id:
                    db_id = plc_idx_to_db_id[plc_idx]
                    w = db.node_size_x[db_id]
                    h = db.node_size_y[db_id]
                    return db_id, float(w / 2.0), float(h / 2.0)

                if plc_idx in pin_indices:
                    pin_name = _get_name(modules[plc_idx])
                    pin_name_norm = _normalize_name(pin_name)
                    return hard_pin_info.get(pin_name, hard_pin_info.get(pin_name_norm))

            return None

        s = _normalize_name(endpoint)

        # Numeric string endpoint.
        if s.isdigit():
            idx = int(s)
            return resolve_endpoint(idx)

        # Exact hard macro pin.
        if s in hard_pin_info:
            return hard_pin_info[s]

        # Exact physical node.
        if s in physical_name_to_db_id:
            db_id = physical_name_to_db_id[s]
            w = db.node_size_x[db_id]
            h = db.node_size_y[db_id]
            return db_id, float(w / 2.0), float(h / 2.0)

        # macro/pin style: fallback to parent physical node.
        if "/" in s:
            parent = s.split("/")[0]
            if parent in physical_name_to_db_id:
                db_id = physical_name_to_db_id[parent]
                w = db.node_size_x[db_id]
                h = db.node_size_y[db_id]
                return db_id, float(w / 2.0), float(h / 2.0)

        return None

    # ============================================================
    # 4. Nets / pins.
    # ============================================================
    db.net_name2id_map = {}
    db.net_names = []
    db.net_weights = []

    db.net2pin_map = []
    db.node2pin_map = [[] for _ in range(db.num_physical_nodes)]

    db.pin2node_map = []
    db.pin2net_map = []
    db.pin_direct = []
    db.pin_offset_x = []
    db.pin_offset_y = []

    skipped_endpoint = 0
    skipped_net_degree = 0
    unresolved_examples = []

    pin_id = 0
    net_id = 0

    for raw_net_id, (key, values) in enumerate(raw_plc.nets.items()):
        values = list(values)

        # Candidate A: key is driver endpoint.
        cand_a = [key] + values

        # Candidate B: key is net name, values are all endpoints.
        cand_b = values

        def resolve_count(cand):
            cnt = 0
            for e in cand:
                if resolve_endpoint(e) is not None:
                    cnt += 1
            return cnt

        cnt_a = resolve_count(cand_a)
        cnt_b = resolve_count(cand_b)

        # Automatically decide whether key is driver endpoint or net name.
        if str(key).startswith("net") and resolve_endpoint(key) is None:
            endpoint_names = cand_b
            net_name = str(key)
        else:
            if cnt_a >= cnt_b:
                endpoint_names = cand_a
                net_name = str(key)
            else:
                endpoint_names = cand_b
                net_name = str(key)

        pin_records = []

        for endpoint_pos, endpoint in enumerate(endpoint_names):
            resolved = resolve_endpoint(endpoint)

            if resolved is None:
                skipped_endpoint += 1
                if len(unresolved_examples) < 80:
                    unresolved_examples.append(endpoint)
                continue

            parent_db_id, offset_x, offset_y = resolved

            pin_records.append(
                {
                    "parent_db_id": int(parent_db_id),
                    "offset_x": float(offset_x),
                    "offset_y": float(offset_y),
                    "direct": b"OUTPUT" if endpoint_pos == 0 else b"INPUT",
                }
            )

        # Remove duplicate endpoint inside a net.
        unique_records = []
        seen = set()

        for rec in pin_records:
            sig = (
                rec["parent_db_id"],
                round(rec["offset_x"], 8),
                round(rec["offset_y"], 8),
            )
            if sig in seen:
                continue
            seen.add(sig)
            unique_records.append(rec)

        if len(unique_records) < 2:
            skipped_net_degree += 1
            continue

        if net_name in db.net_name2id_map:
            net_name = f"{net_name}__{raw_net_id}"

        db.net_name2id_map[net_name] = net_id
        db.net_names.append(net_name)
        db.net_weights.append(1.0)

        pins_of_this_net = []

        for rec in unique_records:
            db.pin2net_map.append(net_id)
            db.pin2node_map.append(rec["parent_db_id"])
            db.pin_direct.append(rec["direct"])
            db.pin_offset_x.append(rec["offset_x"])
            db.pin_offset_y.append(rec["offset_y"])

            db.node2pin_map[rec["parent_db_id"]].append(pin_id)
            pins_of_this_net.append(pin_id)

            pin_id += 1

        db.net2pin_map.append(pins_of_this_net)
        net_id += 1

    db.net_names = _to_bytes_array(db.net_names)
    db.net_weights = np.asarray(db.net_weights, dtype=db.dtype)

    db.pin2node_map = np.asarray(db.pin2node_map, dtype=np.int32)
    db.pin2net_map = np.asarray(db.pin2net_map, dtype=np.int32)
    db.pin_direct = np.asarray(db.pin_direct, dtype=np.bytes_)
    db.pin_offset_x = np.asarray(db.pin_offset_x, dtype=db.dtype)
    db.pin_offset_y = np.asarray(db.pin_offset_y, dtype=db.dtype)

    db.net2pin_map = np.array(
        [np.asarray(x, dtype=np.int32) for x in db.net2pin_map],
        dtype=object,
    )
    db.node2pin_map = np.array(
        [np.asarray(x, dtype=np.int32) for x in db.node2pin_map],
        dtype=object,
    )

    db.flat_net2pin_map, db.flat_net2pin_start_map = _flatten_nested_map(db.net2pin_map)
    db.flat_node2pin_map, db.flat_node2pin_start_map = _flatten_nested_map(db.node2pin_map)

    # ============================================================
    # 5. Compatibility fields.
    # ============================================================
    db.node2orig_node_map = np.arange(db.num_physical_nodes, dtype=np.int32)

    db.pin_names = np.array(
        [f"pin_{i}".encode("utf-8") for i in range(len(db.pin2net_map))],
        dtype=np.bytes_,
    )
    db.pin_name2id_map = {db.pin_names[i]: i for i in range(len(db.pin_names))}

    db.net_weight_deltas = np.zeros(len(db.net_names), dtype=db.dtype)
    db.net_criticality = np.zeros(len(db.net_names), dtype=db.dtype)
    db.net_criticality_deltas = np.zeros(len(db.net_names), dtype=db.dtype)

    db.regions = []
    db.flat_region_boxes = np.array([], dtype=db.dtype)
    db.flat_region_boxes_start = np.array([0], dtype=np.int32)
    db.node2fence_region_map = np.array([], dtype=np.int32)

    # Routing defaults.
    db.routing_grid_xl = db.xl
    db.routing_grid_yl = db.yl
    db.routing_grid_xh = db.xh
    db.routing_grid_yh = db.yh

    db.num_routing_grids_x = getattr(params, "route_num_bins_x", grid_cols)
    db.num_routing_grids_y = getattr(params, "route_num_bins_y", grid_rows)
    db.num_routing_layers = 1

    db.unit_horizontal_capacity = getattr(params, "unit_horizontal_capacity", 1.0)
    db.unit_vertical_capacity = getattr(params, "unit_vertical_capacity", 1.0)
    db.unit_horizontal_capacities = np.array([db.unit_horizontal_capacity], dtype=db.dtype)
    db.unit_vertical_capacities = np.array([db.unit_vertical_capacity], dtype=db.dtype)

    db.initial_horizontal_demand_map = np.zeros(
        (int(db.num_routing_grids_x), int(db.num_routing_grids_y)),
        dtype=db.dtype,
    )
    db.initial_vertical_demand_map = np.zeros(
        (int(db.num_routing_grids_x), int(db.num_routing_grids_y)),
        dtype=db.dtype,
    )

    db.max_net_weight = np.float64(getattr(params, "max_net_weight", 1.0))

    db.num_movable_pins = int(np.sum(db.pin2node_map < db.num_movable_nodes))

    # ============================================================
    # 6. Initialize derived fields manually.
    # ============================================================
    _compute_area_and_macro_info(db, params)

    # Scale before bins, same as original initialize().
    _scale_placedb_like_dreamplace(db, params)

    _initialize_bins(db, params, default_bins_x=grid_cols, default_bins_y=grid_rows)

    if db.num_filler_nodes is None:
        db.num_filler_nodes = 0

    if db.total_filler_node_area is None:
        db.total_filler_node_area = 0.0

    # ============================================================
    # 7. Debug.
    # ============================================================
    if debug:
        print("========== DIRECT PLC -> FULLY INITIALIZED PlaceDB ==========")
        print("canvas:", (db.xl, db.yl, db.xh, db.yh))
        print("grid:", (grid_cols, grid_rows))
        print("num_physical_nodes:", db.num_physical_nodes)
        print("num_movable_nodes:", db.num_movable_nodes)
        print("num_terminals:", db.num_terminals)
        print("num_terminal_NIs:", db.num_terminal_NIs)
        print("num_filler_nodes:", db.num_filler_nodes)
        print("num_nodes:", db.num_nodes)
        print("num_nets:", db.num_nets)
        print("num_pins:", db.num_pins)
        print("num_movable_pins:", db.num_movable_pins)
        print("total_movable_node_area:", db.total_movable_node_area)
        print("total_fixed_node_area:", db.total_fixed_node_area)
        print("total_space_area:", db.total_space_area)
        print("bin:", db.num_bins_x, db.num_bins_y, db.bin_size_x, db.bin_size_y)
        print("raw plc nets:", len(raw_plc.nets))
        print("skipped_endpoint:", skipped_endpoint)
        print("skipped_net_degree:", skipped_net_degree)

        print("mapping:")
        print("  len(db.plc_idx_to_db_id):", len(db.plc_idx_to_db_id))
        print("  len(db.db_id_to_plc_idx):", len(db.db_id_to_plc_idx))
        print("  len(db.movable_plc_indices):", len(db.movable_plc_indices))
        print("  len(db.fixed_plc_indices):", len(db.fixed_plc_indices))
        print("  len(db.physical_plc_indices):", len(db.physical_plc_indices))

        if db.num_nets > 0:
            degree = np.array([len(x) for x in db.net2pin_map], dtype=np.int32)
            print("net degree min/max/mean:", degree.min(), degree.max(), degree.mean())

        if unresolved_examples:
            print("unresolved endpoint examples:")
            for x in unresolved_examples[:40]:
                print("  ", repr(x))

        print("=============================================================")

    return db


def debug_direct_placedb_hpwl(db):
    total_hpwl = 0.0
    nonzero = 0
    zero = 0

    x = db.node_x
    y = db.node_y

    for net_id, pins in enumerate(db.net2pin_map):
        pins = np.asarray(pins, dtype=np.int32)
        nodes = db.pin2node_map[pins]

        xs = x[nodes] + db.pin_offset_x[pins]
        ys = y[nodes] + db.pin_offset_y[pins]

        hpwl = (np.max(xs) - np.min(xs)) + (np.max(ys) - np.min(ys))
        hpwl *= db.net_weights[net_id]

        if hpwl > 0:
            nonzero += 1
            total_hpwl += hpwl
        else:
            zero += 1

    print("========== DIRECT PlaceDB HPWL DEBUG ==========")
    print("manual hpwl:", total_hpwl)
    print("nonzero hpwl nets:", nonzero)
    print("zero hpwl nets:", zero)
    print("================================================")

    return total_hpwl

def apply_net_weights_from_file(
    placedb,
    weight_path,
    default_weight=1.0,
    min_weight=1.0,
    verbose=True,
):
    """
    Read case.weight and assign values to placedb.net_weights.

    Expected .weight format:
        net_5213 1
        net_5214 1
        net_5215 2.5

    Args:
        placedb:
            DREAMPlace PlaceDB.
        weight_path:
            Path to current case weight file, e.g.
            /Macro_challenge_2026/benchmarks/ibm02/ibm02.weight
        default_weight:
            Weight for nets not found in the .weight file.
        min_weight:
            Clamp weight to at least this value. Use 1.0 to avoid zero-weight nets.
        verbose:
            Print debug information.

    Return:
        placedb
    """
    from pathlib import Path
    import numpy as np

    weight_path = Path(weight_path)

    if not weight_path.exists():
        print(f"[NET WEIGHT] Weight file not found, use default weights: {weight_path}")
        placedb.net_weights = np.full(
            len(placedb.net_names),
            float(default_weight),
            dtype=placedb.dtype,
        )
        return placedb

    # ------------------------------------------------------------
    # 1. Read weight file.
    # ------------------------------------------------------------
    weight_dict = {}

    with open(weight_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            net_name = parts[0]

            try:
                weight = float(parts[1])
            except ValueError:
                print(
                    f"[NET WEIGHT][WARN] Bad weight at line {line_no}: {line}"
                )
                continue

            if weight < min_weight:
                weight = min_weight

            weight_dict[net_name] = weight

    # ------------------------------------------------------------
    # 2. Initialize all net weights to default.
    # ------------------------------------------------------------
    net_weights = np.full(
        len(placedb.net_names),
        float(default_weight),
        dtype=placedb.dtype,
    )

    matched = 0
    unmatched_in_file = 0

    # placedb.net_name2id_map should be str -> int.
    net_name2id = placedb.net_name2id_map

    for net_name, weight in weight_dict.items():
        if net_name in net_name2id:
            net_id = net_name2id[net_name]
            net_weights[net_id] = weight
            matched += 1
        else:
            unmatched_in_file += 1

    placedb.net_weights = net_weights

    # ------------------------------------------------------------
    # 3. Also keep compatibility fields consistent.
    # ------------------------------------------------------------
    if not hasattr(placedb, "net_weight_deltas") or placedb.net_weight_deltas is None:
        placedb.net_weight_deltas = np.zeros(
            len(placedb.net_names),
            dtype=placedb.dtype,
        )

    if verbose:
        print("========== APPLY NET WEIGHTS ==========")
        print("weight file:", weight_path)
        print("weights in file:", len(weight_dict))
        print("placedb nets:", len(placedb.net_names))
        print("matched nets:", matched)
        print("unmatched weight names:", unmatched_in_file)
        print(
            "net_weights min/max/mean:",
            float(np.min(placedb.net_weights)) if len(placedb.net_weights) else None,
            float(np.max(placedb.net_weights)) if len(placedb.net_weights) else None,
            float(np.mean(placedb.net_weights)) if len(placedb.net_weights) else None,
        )
        print("#zero net_weights:", int(np.sum(placedb.net_weights == 0)))
        print("=======================================")

    return placedb

def extract_macro_placement_from_placedb(placedb, benchmark, raw_plc, params):
    """
    Extract final macro center positions from DREAMPlace PlaceDB.

    Return:
        placement: torch.Tensor, shape [benchmark.num_macros, 2]

    Macro order:
        hard macros first, then soft macros.

    Important:
        DREAMPlace stores lower-left coordinates.
        benchmark / PLC expects center coordinates.
    """

    import numpy as np
    import torch

    # Prefer unscaled final coordinates saved by PlaceDB.apply().
    if hasattr(placedb, "node_x_unscaled") and hasattr(placedb, "node_y_unscaled"):
        node_x_ll = np.asarray(placedb.node_x_unscaled)
        node_y_ll = np.asarray(placedb.node_y_unscaled)
        already_unscaled = True
    else:
        node_x_ll = np.asarray(placedb.node_x)
        node_y_ll = np.asarray(placedb.node_y)
        already_unscaled = False

    macro_plc_indices = list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices)

    placement_list = []

    for plc_idx in macro_plc_indices:
        if plc_idx not in placedb.plc_idx_to_db_id:
            raise KeyError(
                f"PLC macro index {plc_idx} is not found in placedb.plc_idx_to_db_id. "
                "Please check whether this macro was included in physical_plc_indices."
            )

        db_id = placedb.plc_idx_to_db_id[plc_idx]

        node = raw_plc.modules_w_pins[plc_idx]
        w, h = _get_size(node)

        lx = float(node_x_ll[db_id])
        ly = float(node_y_ll[db_id])

        if not already_unscaled:
            scale = float(params.scale_factor)
            shift_x, shift_y = params.shift_factor
            lx = lx / scale + shift_x
            ly = ly / scale + shift_y

        # Convert lower-left to center.
        cx = lx + w / 2.0
        cy = ly + h / 2.0

        placement_list.append([cx, cy])

    placement = torch.tensor(placement_list, dtype=torch.float32)

    # Respect fixed macros.
    fixed_mask = benchmark.macro_fixed
    placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

    return placement