from __future__ import annotations

import torch

from waymo_open_dataset.protos import map_pb2

from . import map_metric_features

# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10

# Match the road-edge heuristic to limit padding and reduce the number of lane
# segments considered per query.
_LANE_POLYLINE_SEGMENT_LENGTH = 50
_TOP_K_NEAREST_LANE_CHUNKS = 35

# Keep the query chunk bounded so the nearest-lane search fits comfortably on
# GPU even when we evaluate multiple rollouts together.
_MAX_SEGMENT_DISTANCE_ELEMS = 25_000_000


def compute_red_light_violation(
    *,
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    lane_polylines: list,
    lane_ids: list[int],
    traffic_signals: list[list[map_pb2.TrafficSignalLaneState]],
) -> torch.Tensor:
    """Computes red light violations for each of the evaluated objects."""
    if not lane_polylines:
        raise ValueError('Missing lanes.')
    if not traffic_signals:
        raise ValueError('Missing traffic signals.')
    if len(lane_polylines) != len(lane_ids):
        raise ValueError('Inconsistent number of lane polylines and lane ids.')

    squeeze_rollout_dim = center_x.ndim == 2
    if squeeze_rollout_dim:
        center_x = center_x.unsqueeze(0)
        center_y = center_y.unsqueeze(0)
    elif center_x.ndim != 3:
        raise ValueError(
            'Expected `center_x` and `center_y` to have shape '
            '(num_objects, num_steps) or (num_rollouts, num_objects, num_steps).'
        )

    if center_x.shape != center_y.shape:
        raise ValueError('Expected `center_x` and `center_y` to match.')

    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    elif valid.ndim != 3:
        raise ValueError(
            'Expected `valid` to have shape '
            '(num_objects, num_steps) or (num_rollouts, num_objects, num_steps).'
        )

    _, _, num_steps = valid.shape
    if valid.shape[1:] != center_x.shape[1:]:
        raise ValueError(
            'Expected `valid` to match the object and step dimensions of the trajectories.'
        )
    if valid.shape[0] == 1 and center_x.shape[0] != 1:
        valid = valid.expand(center_x.shape[0], -1, -1)
    elif valid.shape[0] != center_x.shape[0]:
        raise ValueError(
            'Expected `valid` rollout dimension to be 1 or match the trajectories.'
        )
    num_rollouts = center_x.shape[0]

    if len(traffic_signals) != num_steps:
        raise ValueError(
            'Expected `traffic_signals` length to match the trajectory step count.'
        )

    device = center_x.device
    evaluated_object_indices = torch.where(evaluated_object_mask)[0]

    xy = torch.stack([center_x, center_y], dim=-1)[:, evaluated_object_indices]
    valid = valid[:, evaluated_object_indices]
    _, num_objects, _, _ = xy.shape

    lane_tensor, lane_ids_tensor, lane_segment_valid = _tensorize_lane_polylines(
        lane_polylines,
        lane_ids,
        seg_length=_LANE_POLYLINE_SEGMENT_LENGTH,
    )
    lane_tensor = lane_tensor.to(device)
    lane_ids_tensor = lane_ids_tensor.to(device)
    lane_segment_valid = lane_segment_valid.to(device)

    xy_flat = xy.reshape(-1, 2)
    nearest_lane_segment_index = _get_nearest_lane_segment_index(
        xy=xy_flat,
        lane_xyz_valid=lane_tensor.unsqueeze(0),
        segment_valid=lane_segment_valid,
    )
    nearest_lane_index = nearest_lane_segment_index[:, 0]
    current_lane_id_flat = lane_ids_tensor[nearest_lane_index]
    current_lane_id = current_lane_id_flat.reshape(num_rollouts, num_objects, num_steps)

    ts_lane_id, ts_state, ts_stop_point = _tensorize_traffic_signals(
        traffic_signals, device=device
    )
    num_traffic_signals = ts_lane_id.shape[1]

    ts_match = current_lane_id.unsqueeze(-1) == ts_lane_id.unsqueeze(0).unsqueeze(0)
    ts_is_stop = (
        (ts_state == map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP)
        | (ts_state == map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP)
    )

    ts_match_lane = lane_ids_tensor.view(1, 1, -1) == ts_lane_id.unsqueeze(-1)
    ts_lane_valid = torch.any(ts_match_lane, dim=-1)
    ts_stop_point_nearest = _get_nearest_lane_segment_index(
        xy=ts_stop_point.reshape(-1, 2),
        lane_xyz_valid=lane_tensor.unsqueeze(0),
        segment_valid=lane_segment_valid,
        lane_mask=ts_match_lane.reshape(-1, lane_tensor.shape[0]),
    )
    ts_lane_index = ts_stop_point_nearest[:, 0].reshape(num_steps, num_traffic_signals)
    ts_segments = lane_tensor[ts_lane_index]
    ts_stop_point_segment_index_flat = ts_stop_point_nearest[:, 1]
    ts_stop_point_segment_index = ts_stop_point_segment_index_flat.reshape(
        num_steps, num_traffic_signals
    )
    ts_stop_point_segment_indices = torch.stack(
        [ts_stop_point_segment_index, ts_stop_point_segment_index + 1],
        dim=-1,
    )
    ts_stop_point_segment = torch.gather(
        ts_segments,
        2,
        ts_stop_point_segment_indices.unsqueeze(-1).expand(-1, -1, -1, 4),
    )

    start_to_end = ts_stop_point_segment[..., 1, :2] - ts_stop_point_segment[..., 0, :2]
    stop_point_segment_length2 = map_metric_features.dot_product_2d(
        start_to_end, start_to_end
    )
    start_to_stop_point = ts_stop_point - ts_stop_point_segment[..., 0, :2]
    start_to_xy = (
        xy.unsqueeze(-2) - ts_stop_point_segment.unsqueeze(0).unsqueeze(0)[..., 0, :2]
    )

    stop_point_rel_t = torch.div(
        map_metric_features.dot_product_2d(start_to_stop_point, start_to_end),
        stop_point_segment_length2.clamp(min=1e-10),
    )
    object_rel_t = torch.div(
        map_metric_features.dot_product_2d(
            start_to_xy, start_to_end.unsqueeze(0).unsqueeze(0)
        ),
        stop_point_segment_length2.unsqueeze(0).unsqueeze(0).clamp(min=1e-10),
    )
    object_behind_stop_point = object_rel_t < stop_point_rel_t.unsqueeze(0).unsqueeze(0)
    object_ahead_stop_point = object_rel_t > stop_point_rel_t.unsqueeze(0).unsqueeze(0)
    object_crossed_stop_point = torch.logical_and(
        object_behind_stop_point[:, :, :-1],
        object_ahead_stop_point[:, :, 1:],
    )
    object_crossed_stop_point = torch.cat(
        [torch.zeros_like(object_crossed_stop_point[:, :, 0:1]), object_crossed_stop_point],
        dim=2,
    )

    red_light_violation = torch.any(
        valid.unsqueeze(-1)
        & ts_match
        & ts_is_stop.unsqueeze(0).unsqueeze(0)
        & ts_lane_valid.unsqueeze(0).unsqueeze(0)
        & object_crossed_stop_point,
        dim=-1,
    )

    if squeeze_rollout_dim:
        red_light_violation = red_light_violation[0]
    return red_light_violation


def _get_nearest_lane_segment_index(
    *,
    xy: torch.Tensor,
    lane_xyz_valid: torch.Tensor,
    segment_valid: torch.Tensor | None = None,
    lane_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the index of the nearest lane segment in 2D space."""
    num_points = xy.shape[0]
    if lane_xyz_valid.shape[0] not in (1, num_points):
        raise ValueError(
            '`lane_xyz_valid` must have a batch dimension of 1 or match `xy`.'
        )
    num_lanes = lane_xyz_valid.shape[1]
    if lane_mask is not None and lane_mask.shape not in ((1, num_lanes), (num_points, num_lanes)):
        raise ValueError(
            '`lane_mask` must have shape (1, num_lanes) or (num_points, num_lanes).'
        )
    num_segments = max(lane_xyz_valid.shape[2] - 1, 1)
    if segment_valid is not None and segment_valid.shape != (num_lanes, num_segments):
        raise ValueError(
            '`segment_valid` must have shape (num_lanes, num_segments).'
        )

    elems_per_point = max(min(_TOP_K_NEAREST_LANE_CHUNKS, num_lanes) * num_segments, 1)
    chunk_size = max(1, _MAX_SEGMENT_DISTANCE_ELEMS // elems_per_point)
    chunk_size = min(chunk_size, num_points)

    nearest_segment_indices = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        lane_xyz_valid_chunk = lane_xyz_valid
        if lane_xyz_valid.shape[0] != 1:
            lane_xyz_valid_chunk = lane_xyz_valid[start:end]
        elif end - start != 1:
            lane_xyz_valid_chunk = lane_xyz_valid.expand(end - start, -1, -1, -1)

        lane_mask_chunk = None
        if lane_mask is not None:
            lane_mask_chunk = lane_mask
            if lane_mask.shape[0] != 1:
                lane_mask_chunk = lane_mask[start:end]
            elif end - start != 1:
                lane_mask_chunk = lane_mask.expand(end - start, -1)

        segment_valid_chunk = None
        if segment_valid is not None:
            segment_valid_chunk = segment_valid.unsqueeze(0).expand(end - start, -1, -1)

        lane_point_valid = lane_xyz_valid_chunk[:, :, :, 3].bool()
        first_valid_idx, last_valid_idx = _find_first_and_last_true(
            lane_point_valid.reshape(-1, lane_point_valid.shape[-1])
        )
        first_valid_idx = first_valid_idx.reshape(end - start, num_lanes)
        last_valid_idx = last_valid_idx.reshape(end - start, num_lanes)
        lane_range = torch.arange(num_lanes, device=xy.device).unsqueeze(0).expand(end - start, -1)
        batch_range = torch.arange(end - start, device=xy.device).unsqueeze(1).expand(-1, num_lanes)
        first_valid_points = lane_xyz_valid_chunk[batch_range, lane_range, first_valid_idx, :2]
        last_valid_points = lane_xyz_valid_chunk[batch_range, lane_range, last_valid_idx, :2]
        middle_valid_idx = (first_valid_idx + last_valid_idx) // 2
        middle_valid_points = lane_xyz_valid_chunk[batch_range, lane_range, middle_valid_idx, :2]
        test_points = torch.stack(
            [first_valid_points, last_valid_points, middle_valid_points],
            dim=2,
        )
        rough_distances = (
            xy[start:end, None, None] - test_points
        ).norm(dim=-1).min(dim=-1).values
        if lane_mask_chunk is not None:
            rough_distances = torch.where(
                lane_mask_chunk,
                rough_distances,
                torch.full_like(rough_distances, EXTREMELY_LARGE_DISTANCE),
            )
        top_k = min(_TOP_K_NEAREST_LANE_CHUNKS, num_lanes)
        topk_idx = rough_distances.topk(
            k=top_k,
            dim=-1,
            largest=False,
            sorted=False,
        ).indices
        gather_index = topk_idx[:, :, None, None].expand(
            -1,
            -1,
            lane_xyz_valid_chunk.shape[2],
            lane_xyz_valid_chunk.shape[3],
        )
        lane_xyz_valid_chunk = torch.gather(lane_xyz_valid_chunk, 1, gather_index)
        if segment_valid_chunk is not None:
            segment_gather_index = topk_idx[:, :, None].expand(-1, -1, num_segments)
            segment_valid_chunk = torch.gather(segment_valid_chunk, 1, segment_gather_index)
        if lane_mask_chunk is not None:
            topk_mask = torch.gather(lane_mask_chunk, 1, topk_idx)
            lane_xyz_valid_chunk = torch.where(
                topk_mask[:, :, None, None],
                lane_xyz_valid_chunk,
                torch.zeros_like(lane_xyz_valid_chunk),
            )
            if segment_valid_chunk is not None:
                segment_valid_chunk = topk_mask[:, :, None] & segment_valid_chunk

        lane_xy = lane_xyz_valid_chunk[:, :, :, :2]
        lane_valid = lane_xyz_valid_chunk[:, :, :, 3].bool()

        segment_start_xy = lane_xy[:, :, :-1]
        segment_end_xy = lane_xy[:, :, 1:]
        start_to_point = xy[start:end, None, None] - segment_start_xy
        start_to_end = segment_end_xy - segment_start_xy

        rel_t = torch.div(
            map_metric_features.dot_product_2d(start_to_point, start_to_end),
            map_metric_features.dot_product_2d(start_to_end, start_to_end).clamp(min=1e-10),
        )
        clipped_rel_t = rel_t.clamp(0.0, 1.0)

        # Match the official TensorFlow implementation exactly here, including
        # the distance expression used for segment association.
        distance_to_segment = torch.linalg.norm(
            start_to_point + start_to_end * clipped_rel_t.unsqueeze(-1),
            dim=-1,
        )
        if segment_valid_chunk is None:
            segment_valid_chunk = lane_valid[:, :, :-1]
        distance_to_segment = torch.where(
            segment_valid_chunk,
            distance_to_segment,
            torch.full_like(distance_to_segment, EXTREMELY_LARGE_DISTANCE),
        )
        nearest_segment_2d_index = _argmin_2d(distance_to_segment)
        nearest_lane_index = topk_idx[
            torch.arange(end - start, device=xy.device),
            nearest_segment_2d_index[:, 0],
        ]
        nearest_segment_indices.append(
            torch.stack([nearest_lane_index, nearest_segment_2d_index[:, 1]], dim=1)
        )

    return torch.cat(nearest_segment_indices, dim=0)


def _argmin_2d(t: torch.Tensor) -> torch.Tensor:
    """Finds the 2D indices of the minimum element in a 3D tensor."""
    flat_indices = torch.argmin(t.reshape(t.shape[0], -1), dim=1)
    num_cols = t.shape[2]
    cols = flat_indices % num_cols
    rows = flat_indices // num_cols
    return torch.stack([rows, cols], dim=1)


def _find_first_and_last_true(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the first and last valid point index for each lane chunk."""
    num_rows, num_cols = mask.shape
    int_mask = mask.to(torch.uint8)
    first_indices = torch.argmax(int_mask, dim=1)
    flipped_mask = torch.flip(int_mask, dims=[1])
    flipped_first_indices = torch.argmax(flipped_mask, dim=1)
    last_indices = (num_cols - 1) - flipped_first_indices
    valid_rows = mask.any(dim=1)
    first_indices = torch.where(valid_rows, first_indices, torch.zeros_like(first_indices))
    last_indices = torch.where(valid_rows, last_indices, torch.zeros_like(last_indices))
    return first_indices, last_indices


def _tensorize_lane_polylines(
    polylines: list,
    ids: list[int],
    seg_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensorizes lane polylines while preserving the official tail-padding bug."""
    if len(polylines) != len(ids):
        raise ValueError('Inconsistent number of polylines and ids.')

    polyline_tensors = []
    feature_ids = []
    max_polyline_length = 0
    for polyline, feature_id in zip(polylines, ids):
        if len(polyline) < 2:
            continue
        point_tensor = torch.tensor(
            [[map_point.x, map_point.y, map_point.z, 1.0] for map_point in polyline],
            dtype=torch.float32,
        )
        polyline_tensors.append(point_tensor)
        feature_ids.append(feature_id)
        max_polyline_length = max(max_polyline_length, point_tensor.shape[0])

    if not polyline_tensors:
        return (
            torch.empty((0, 0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.int32),
            torch.empty((0, 0), dtype=torch.bool),
        )

    effective_seg_length = max_polyline_length
    if seg_length is not None:
        effective_seg_length = min(seg_length, max_polyline_length)
        if effective_seg_length < 2:
            raise ValueError('seg_length must be at least 2.')

    stacked_polylines = []
    chunk_ids = []
    segment_validity = []
    for polyline_tensor, feature_id in zip(polyline_tensors, feature_ids):
        line_length = polyline_tensor.shape[0]
        is_full_length = line_length == max_polyline_length
        for sub_idx, sub_num, start_idx, end_idx in map_metric_features._iter_polyline_chunks(
            line_length, effective_seg_length, is_full_length
        ):
            chunk_length = end_idx - start_idx
            padded_polyline = torch.zeros(
                (effective_seg_length + 1, 4),
                dtype=torch.float32,
                device=polyline_tensor.device,
            )
            padded_polyline[:chunk_length] = polyline_tensor[start_idx:end_idx]
            stacked_polylines.append(padded_polyline)
            chunk_ids.append(feature_id)

            # The official implementation pads every non-max lane once and then
            # incorrectly keeps the tail segment `[last_real_point -> 0]` valid.
            # With chunking we need to preserve that single virtual segment only
            # for the terminal chunk of lanes shorter than the global max.
            segment_valid = torch.zeros(
                (effective_seg_length,),
                dtype=torch.bool,
                device=polyline_tensor.device,
            )
            segment_valid[: max(chunk_length - 1, 0)] = True
            if sub_idx == sub_num - 1 and not is_full_length:
                segment_valid[chunk_length - 1] = True
            segment_validity.append(segment_valid)

    return (
        torch.stack(stacked_polylines, dim=0),
        torch.tensor(chunk_ids, dtype=torch.int32),
        torch.stack(segment_validity, dim=0),
    )

def _tensorize_traffic_signals(
    traffic_signals: list[list[map_pb2.TrafficSignalLaneState]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts a per-time, per-lane list of traffic signal states to tensors."""
    num_steps = len(traffic_signals)
    all_lane_ids = []
    for signals_at_t in traffic_signals:
        for state in signals_at_t:
            all_lane_ids.append(state.lane)
    all_lane_ids = list(set(all_lane_ids))
    num_tl_lanes = len(all_lane_ids)
    if num_tl_lanes == 0:
        all_lane_ids = [-1]
        num_tl_lanes = 1

    lane_ids = torch.tensor(all_lane_ids, dtype=torch.int32, device=device)
    lane_ids = lane_ids.unsqueeze(0).repeat(num_steps, 1)
    states = torch.zeros((num_steps, num_tl_lanes), dtype=torch.int32, device=device)
    stop_points = torch.zeros((num_steps, num_tl_lanes, 2), dtype=torch.float32, device=device)
    lane_id_to_index = {lane_id: idx for idx, lane_id in enumerate(all_lane_ids)}
    for t, signals_at_t in enumerate(traffic_signals):
        for state in signals_at_t:
            lane_index = lane_id_to_index[state.lane]
            states[t, lane_index] = state.state
            stop_points[t, lane_index, 0] = state.stop_point.x
            stop_points[t, lane_index, 1] = state.stop_point.y
    return lane_ids, states, stop_points
