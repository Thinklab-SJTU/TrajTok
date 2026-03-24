from typing import Optional, Sequence

import torch

import time
_distance_computation_total_time = 0.0
_distance_computation_call_count = 0
# Constant distance to apply when distances are invalid. This will avoid the
# propagation of nans and should be reduced out when taking the minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Off-road threshold, i.e. smallest distance away from the road edge that is
# considered to be a off-road.
OFFROAD_DISTANCE_THRESHOLD = 0.0

# How close the start and end point of a map feature need to be for the feature
# to be considered cyclic, in m^2.
_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0
# Scaling factor for vertical distances used when finding the closest segment to
# a query point. This prevents wrong associations in cases with under- and
# over-passes.
_Z_STRETCH_FACTOR = 3.0


def dot_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the dot product between two 2D vectors.

    Args:
        a: A tensor of shape (..., 2) containing the first 2D vector.
        b: A tensor of shape (..., 2) containing the second 2D vector.

    Returns:
        A tensor of shape (...) containing the dot product between the vectors.
    """
    return torch.sum(a * b, dim=-1)


def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the z-component of the cross product between two 2D vectors.

    Args:
        a: A tensor of shape (..., 2) containing the first 2D vector.
        b: A tensor of shape (..., 2) containing the second 2D vector.

    Returns:
        A tensor of shape (...) containing the z-component of the cross product
        between the vectors.
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def compute_distance_to_road_edge(
        *,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        evaluated_object_mask: torch.Tensor,
        road_edge_polylines: Sequence[torch.Tensor]
    ) -> torch.Tensor:
    """Computes the distance to the road edge for each of the evaluated objects.

    Args:
        boxes: A float Tensor of shape (num_rollouts, num_objects, num_steps, [x,y,z,l,w,h,heading])
        valid: A boolean Tensor of shape (num_objects, num_steps) containing the
            validity of the objects over time.
        evaluated_object_mask: A boolean tensor of shape (num_objects), indicating
            whether each object should be considered part of the "evaluation set".
        road_edge_polylines: A sequence of polylines, each defined as a sequence of
            3d points with x, y, and z-coordinates. The polylines should be oriented
            such that port side is on-road and starboard side is off-road, a.k.a
            counterclockwise winding order.

    Returns:
        A tensor of shape (num_rollouts, num_evaluated_objects, num_steps), containing the
        distance to the road edge, for each timestep and for all the objects
        to be evaluated, as specified by `evaluated_object_mask`.

    Raises:
        ValueError: When the `road_edge_polylines` is empty, i.e. there is no map
            information in the Scenario.
    """
    if not road_edge_polylines:
        raise ValueError('Missing road edges.')
    num_rollouts, num_objects, num_steps, num_features = boxes.shape
    boxes = boxes.reshape(num_rollouts * num_objects * num_steps, num_features)
    # Compute box corners using `box_utils`, and take the xyz coords of the bottom
    # corners.
    box_corners = get_upright_3d_box_corners(boxes)[:, :4]
    box_corners = box_corners.reshape(num_rollouts, num_objects, num_steps, 4, 3)

    # Gather objects in the evaluation set
    # `eval_corners` shape: (num_rollouts, num_evaluated_objects, num_steps, 4, 3).
    eval_corners = box_corners[:, evaluated_object_mask]
    num_eval_objects = eval_corners.shape[1]

    # Flatten query points.
    # `flat_eval_corners` shape: (num_rollouts * num_evaluated_objects * num_steps * 4, 3).
    flat_eval_corners = eval_corners.reshape(-1, 3)

    # Tensorize road edges.
    polylines_tensor, use_left_neighbor, use_right_neighbor, left_neighbors, right_neighbors = _tensorize_polylines(road_edge_polylines, seg_length=50)
    #is_polyline_cyclic = _check_polyline_cycles(splited_polylines)

    # Compute distances for all query points.
    # `corner_distance_to_road_edge` shape: (num_rollouts * num_evaluated_objects * num_steps * 4).
    corner_distance_to_road_edge = _compute_signed_distance_to_polylines(
            xyzs=flat_eval_corners, polylines=polylines_tensor, use_left_neighbor=use_left_neighbor, use_right_neighbor=use_right_neighbor, left_neighbors=left_neighbors, right_neighbors=right_neighbors, z_stretch=_Z_STRETCH_FACTOR
    )
    # `corner_distance_to_road_edge` shape: (num_rollouts, num_evaluated_objects, num_steps, 4).
    corner_distance_to_road_edge = corner_distance_to_road_edge.reshape(
            num_rollouts, num_eval_objects, num_steps, 4
    )

    # Reduce to most off-road corner.
    # `signed_distances` shape: (num_rollouts, num_evaluated_objects, num_steps).
    signed_distances = torch.max(corner_distance_to_road_edge, dim=-1)[0]

    # Mask out invalid boxes.
    eval_validity = valid[evaluated_object_mask]
    eval_validity = eval_validity.unsqueeze(0).expand(num_rollouts, -1, -1)

    return torch.where(eval_validity, signed_distances, -EXTREMELY_LARGE_DISTANCE)

def _iter_polyline_chunks(
        line_length: int,
        seg_length: int,
        is_full_length: bool,
):
    """Yields chunk boundaries following the existing map splitting heuristic."""
    sub_num = (line_length - 1) // (seg_length - 1)
    if (line_length - 1) % (seg_length - 1) != 0:
        sub_num += 1

    for sub_idx in range(sub_num):
        start_idx = max(sub_idx * (seg_length - 1), 0)
        end_idx = min(start_idx + seg_length, line_length)
        if sub_idx == sub_num - 1 and (is_full_length or (end_idx - start_idx) < 5):
            start_idx = max(line_length - seg_length, 0)
            end_idx = line_length
        yield sub_idx, sub_num, start_idx, end_idx


def tensorize_polylines(
        polylines: list,
        ids: list[int] | None = None,
        seg_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacks a sequence of polylines into a tensor.

    Args:
            polylines: A sequence of Polyline objects.
            ids: A sequence of integer ids for each polyline in `polylines`. If None,
                    the all zeros ids are returned.

    Returns:
            A float tensor with shape (num_polylines, max_length, 4) containing xyz
                    coordinates and a validity flag for all points in the polylines.
                    When `seg_length` is set, long polylines are split into chunks of
                    roughly equal length using the same heuristic as road-edge metrics.
            A int tensor with shape (num_polylines) containing the ids of the polylines.

    Raises:
            ValueError: When the number of polylines and ids are inconsistent.
    """
    if ids is None:
            ids = [0] * len(polylines)
    elif len(polylines) != len(ids):
            raise ValueError('Inconsistent number of polylines and ids.')

    polyline_tensors = []
    feature_ids = []

    max_length = 0
    for polyline, feature_id in zip(polylines, ids):
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        max_length = max(max_length, len(polyline))
        polyline_tensors.append(
            # shape: (num_segments+1, 4: x,y,z,valid)
            torch.tensor([
                [map_point.x, map_point.y, map_point.z, 1.0] for map_point in polyline
            ], dtype=torch.float32)
        )
        feature_ids.append(feature_id)

    if not polyline_tensors:  # Handle case where all polylines are degenerate
            return torch.empty((0, 0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int32)

    if seg_length is not None:
            max_polyline_length = max(p.shape[0] for p in polyline_tensors)
            seg_length = min(seg_length, max_polyline_length)
            if seg_length < 2:
                    raise ValueError('seg_length must be at least 2.')

            stacked_polylines = []
            id_tensors = []
            for polyline_tensor, feature_id in zip(polyline_tensors, feature_ids):
                    line_length = polyline_tensor.shape[0]
                    is_full_length = line_length == max_polyline_length
                    for _, _, start_idx, end_idx in _iter_polyline_chunks(
                            line_length, seg_length, is_full_length
                    ):
                            padded_polyline = torch.zeros(
                                    (seg_length, 4),
                                    dtype=torch.float32,
                                    device=polyline_tensor.device,
                            )
                            padded_polyline[:end_idx - start_idx] = polyline_tensor[start_idx:end_idx]
                            stacked_polylines.append(padded_polyline)
                            id_tensors.append(feature_id)

            return torch.stack(stacked_polylines, dim=0), torch.tensor(id_tensors, dtype=torch.int32)

    # Stack polylines with padding
    stacked_polylines = []
    for p in polyline_tensors:
            if p.shape[0] < max_length:
                    padding = torch.zeros((max_length - p.shape[0], 4), dtype=torch.float32)
                    padded_polyline = torch.cat([p, padding], dim=0)
            else:
                    padded_polyline = p
            stacked_polylines.append(padded_polyline)
    
    # shape: (num_polylines, max_length, 4)
    stacked_polylines = torch.stack(stacked_polylines, dim=0)
    
    # shape: (num_polylines)
    return stacked_polylines, torch.tensor(feature_ids, dtype=torch.int32)

def _tensorize_polylines(polylines: Sequence[torch.Tensor], seg_length: int=None) -> torch.Tensor:
    """Stacks a sequence of polylines into a tensor.

    Args:
        polylines: A sequence of Polyline objects.

    Returns:
        A float tensor with shape (num_polylines, max_length, 4) containing xyz
            coordinates and a validity flag for all points in the polylines. Polylines
            are padded with zeros up to the length of the longest one.
    """

    tensorize_polylines = []
    use_left_neighbor = []
    use_right_neighbor = []
    left_neighbors = []
    right_neighbors = []
    max_polyline_length = max([len(polyline) for polyline in polylines])
    if seg_length:
        seg_length = min(seg_length, max_polyline_length)
        for i, polyline in enumerate(polylines):
            if len(polyline) < 2:
                continue
            has_cycle = torch.sum(torch.square(polyline[0] - polyline[-1]), dim=-1) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
            is_full_length = len(polyline) == max_polyline_length
            line_length = len(polyline)
            for sub_idx, sub_num, start_idx, end_idx in _iter_polyline_chunks(
                line_length, seg_length, is_full_length
            ):
                tensorize_polyline = torch.zeros(seg_length, 4, device=polyline.device) 
                tensorize_polyline[0:end_idx-start_idx,0:3] = polyline[start_idx:end_idx]
                tensorize_polyline[0:end_idx-start_idx,3] = 1.0
                left_neighbor = torch.zeros(4, device=polyline.device) 
                right_neighbor = torch.zeros(4, device=polyline.device) 
                if sub_idx == 0:
                    if is_full_length:
                        left_neighbor[0:3] = polyline[-1]
                        left_neighbor[3] = 1.0
                    use_left_neighbor.append(has_cycle)
                else:
                    left_neighbor[0:3] = polyline[start_idx-1]
                    left_neighbor[3] = 1.0
                    use_left_neighbor.append(True)
                if sub_idx == sub_num - 1:
                    if is_full_length:
                        right_neighbor[0:3] = polyline[0]
                        right_neighbor[3] = 1.0
                    use_right_neighbor.append(has_cycle)
                else:
                    right_neighbor[0:3] = polyline[end_idx]
                    right_neighbor[3] = 1.0
                    use_right_neighbor.append(True)
                left_neighbors.append(left_neighbor)
                right_neighbors.append(right_neighbor)
                tensorize_polylines.append(tensorize_polyline)
    else:
        for i, polyline in enumerate(polylines):
            if len(polyline) < 2:
                continue
            has_cycle = torch.sum(torch.square(polyline[0] - polyline[-1]), dim=-1) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2
            tensorize_polyline = torch.zeros(max_polyline_length, 4, device=polyline.device)
            tensorize_polyline[0:len(polyline),3] = 1.  
            tensorize_polyline[0:len(polyline),0:3] = polyline            
            if has_cycle:
                use_left_neighbor.append(True)
                use_right_neighbor.append(True)
            else:
                use_left_neighbor.append(False)
                use_right_neighbor.append(False)
            tensorize_polylines.append(tensorize_polyline)
            right_neighbors.append(tensorize_polyline[0].clone())
            left_neighbors.append(tensorize_polyline[-1].clone())
    return torch.stack(tensorize_polylines, dim=0), torch.tensor(use_left_neighbor, device=polylines[0].device), torch.tensor(use_right_neighbor, device=polylines[0].device), torch.stack(left_neighbors, dim=0), torch.stack(right_neighbors, dim=0)

def find_first_and_last_true(mask):
    m, n = mask.shape
    int_mask = mask.to(torch.uint8)
    first_indices = torch.argmax(int_mask, dim=1)
    flipped_mask = torch.flip(int_mask, dims=[1])
    flipped_first_indices = torch.argmax(flipped_mask, dim=1)
    last_indices = (n - 1) - flipped_first_indices
    valid_rows = mask.any(dim=1)
    first_indices = torch.where(valid_rows, first_indices, torch.tensor(-1, device=mask.device))
    last_indices = torch.where(valid_rows, last_indices, torch.tensor(-1, device=mask.device))
    return first_indices, last_indices


def _check_polyline_cycles(polylines: Sequence[torch.Tensor]) -> torch.Tensor:
    """Checks if given polylines are cyclic and returns the result as a tensor.

    Args:
        polylines: A sequence of Polyline objects.

    Returns:
        A bool tensor with shape (num_polylines) indicating whether each polyline is
        cyclic.
    """
    cycles = []
    for polyline in polylines:
        # Skip degenerate polylines.
        if len(polyline) < 2:
            continue
        cycles.append(torch.sum(torch.square(polyline[0] - polyline[-1]), dim=-1)< _CYCLIC_MAP_FEATURE_TOLERANCE_M2)
    # shape: (num_polylines)
    return torch.stack(cycles, dim=0)

def _compute_signed_distance_to_polylines(
        xyzs: torch.Tensor, #[n_point,3]
        polylines: torch.Tensor, #[n_polyline,n_segment+1,4]
        use_left_neighbor: torch.Tensor | None,
        use_right_neighbor: torch.Tensor | None,
        left_neighbors: torch.Tensor | None,
        right_neighbors: torch.Tensor | None,
        z_stretch: float,
        top_k: int = 35  
    ) -> torch.Tensor:
         
        # polylines = torch.stack(new_polylines)
        num_points = xyzs.shape[0]
        num_polylines = polylines.shape[0]
        num_segments = polylines.shape[1] -1

        first_valid_idx, last_valid_idx = find_first_and_last_true(polylines[:, :, 3].bool())
        range_t = torch.arange(num_polylines, device=polylines.device)
        first_valid_points = polylines[range_t, first_valid_idx, 0:3] 
        last_valid_points = polylines[range_t, last_valid_idx, 0:3]
        middle_points = polylines[range_t, (first_valid_idx + last_valid_idx)//2, 0:3]
        test_points = torch.stack([first_valid_points, last_valid_points, middle_points], dim=1)
        rough_distances = (xyzs[:,None,None,:] - test_points[None, :, :]).norm(dim=-1).min(dim=-1).values
        topk_idx = rough_distances.topk(k=min(top_k, num_polylines), dim=-1, largest=False, sorted=False).indices
        
        polylines_expanded = torch.cat([left_neighbors[:, None, :], polylines, right_neighbors[:, None, :]], dim=1)
        polylines_expanded_topk = polylines_expanded[topk_idx]
        is_point_valid_expanded_topk = polylines_expanded_topk[:, :, :, 3].bool()
        is_segment_valid_expanded_topk = torch.logical_and(is_point_valid_expanded_topk[:, :, :-1], is_point_valid_expanded_topk[:, :, 1:])
        is_segment_valid_topk = is_segment_valid_expanded_topk[:, :,1:-1]

        start_to_point_expanded = xyzs[:, None, None, :3] - polylines_expanded_topk[:, :, :-1, :3]
        start_to_end_expanded = polylines_expanded_topk[:, :, 1:, :3] - polylines_expanded_topk[:, :, :-1, :3]
        start_to_point = start_to_point_expanded[:, :, 1:-1, :]
        start_to_end = start_to_end_expanded[:,:, 1:-1,:] 

        rel_t = torch.div(
            dot_product_2d(
                start_to_point[..., :2], start_to_end[..., :2]
            ),
            dot_product_2d(
                start_to_end[..., :2], start_to_end[..., :2]
            ).clamp(min=1e-10)
        )

        segment_to_point = start_to_point - (
            start_to_end * torch.clamp(rel_t, 0.0, 1.0).unsqueeze(-1)
        )

        stretch_vector = torch.tensor([1.0, 1.0, z_stretch], dtype=torch.float32, device=polylines.device)
        # shape: (num_points, top_k, num_segments)
        distance_to_segment_3d = torch.norm(
            segment_to_point * stretch_vector.view(1, 1, 1, 3), dim=-1).reshape(num_points, -1)
        distance_to_segment_3d = torch.where(
            is_segment_valid_topk.reshape(num_points, -1),
            distance_to_segment_3d,
            EXTREMELY_LARGE_DISTANCE,
        )
        closest_segment_index = torch.argmin(distance_to_segment_3d, dim=-1)
        closest_line_idx = closest_segment_index // num_segments 
        closest_segment_idx_in_line = closest_segment_index % num_segments 
        range_t = torch.arange(num_points, device=xyzs.device)

        closest_distance_to_segment_2d = segment_to_point[range_t, closest_line_idx, closest_segment_idx_in_line, 0:2].norm(dim=-1)
        is_closest_segment_valid = is_segment_valid_topk[range_t, closest_line_idx, closest_segment_idx_in_line]
        distance_2d = torch.where(is_closest_segment_valid, closest_distance_to_segment_2d, EXTREMELY_LARGE_DISTANCE)

        closest_segment_idx_in_line_expanded = closest_segment_index % num_segments + 1

        prior_segment_idx_in_line_expanded = closest_segment_idx_in_line_expanded - 1
        prior_segment_idx_in_line_expanded_cond = prior_segment_idx_in_line_expanded.clone()
        copy_left = (prior_segment_idx_in_line_expanded_cond == 0) & ~use_left_neighbor[topk_idx[range_t, closest_line_idx]]
        prior_segment_idx_in_line_expanded_cond[copy_left] = 1

        next_segment_idx_in_line_expanded = closest_segment_idx_in_line_expanded + 1
        next_segment_idx_in_line_expanded_cond = next_segment_idx_in_line_expanded.clone()
        copy_right = (next_segment_idx_in_line_expanded_cond == num_segments+1) & ~use_right_neighbor[topk_idx[range_t, closest_line_idx]]
        next_segment_idx_in_line_expanded_cond[copy_right] = num_segments

        
        n = torch.sign(
            cross_product_2d(
                start_to_point_expanded[range_t, closest_line_idx, closest_segment_idx_in_line_expanded, :2],
                start_to_end_expanded[range_t, closest_line_idx, closest_segment_idx_in_line_expanded, :2]
            )
        )
        n_prior = torch.sign(
            cross_product_2d(
                start_to_point_expanded[range_t, closest_line_idx, prior_segment_idx_in_line_expanded_cond, :2],
                start_to_end_expanded[range_t, closest_line_idx, prior_segment_idx_in_line_expanded_cond, :2]
            )
        )
        n_next = torch.sign(
            cross_product_2d(
                start_to_point_expanded[range_t, closest_line_idx, next_segment_idx_in_line_expanded_cond, :2],
                start_to_end_expanded[range_t, closest_line_idx, next_segment_idx_in_line_expanded_cond, :2]
            )
        )
        
        is_locally_convex_before = torch.greater(
            cross_product_2d(start_to_end_expanded[range_t, closest_line_idx, prior_segment_idx_in_line_expanded, :2],
                             start_to_end_expanded[range_t, closest_line_idx, closest_segment_idx_in_line_expanded, :2]), 0.0)
        is_locally_convex_after = torch.greater(
            cross_product_2d(start_to_end_expanded[range_t, closest_line_idx, closest_segment_idx_in_line_expanded, :2],
                             start_to_end_expanded[range_t, closest_line_idx, next_segment_idx_in_line_expanded, :2]), 0.0)        

        is_prior_segment_valid = is_segment_valid_expanded_topk[range_t, closest_line_idx, prior_segment_idx_in_line_expanded_cond]
        is_next_segment_valid = is_segment_valid_expanded_topk[range_t, closest_line_idx, next_segment_idx_in_line_expanded_cond]
        sign_if_before = torch.where(is_locally_convex_before, torch.maximum(n, n_prior), torch.minimum(n, n_prior))        
        sign_if_after = torch.where(is_locally_convex_after, torch.maximum(n, n_next), torch.minimum(n, n_next))
        rel_t_closest = rel_t[range_t, closest_line_idx, closest_segment_idx_in_line]

        distance_sign = torch.where(
            (rel_t_closest < 0.0) & is_prior_segment_valid,
            sign_if_before,
            torch.where((rel_t_closest > 1.0) & is_next_segment_valid, sign_if_after, n)
        )

        return distance_sign * distance_2d


def get_upright_3d_box_corners(boxes):
    """Given a set of upright boxes, return its 8 corners.

    Given a set of boxes, returns its 8 corners. The corners are ordered layers
    (bottom, top) first and then counter-clockwise within each layer.

    Args:
        boxes: torch Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
            height, heading].

    Returns:
        corners: torch Tensor [N, 8, 3].
    """
    center_x, center_y, center_z, length, width, height, heading = torch.unbind(
            boxes, dim=-1)

    # [N, 3, 3]
    rotation = get_yaw_rotation(heading)
    # [N, 3]
    translation = torch.stack([center_x, center_y, center_z], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = torch.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
    ], dim=-1).reshape(-1, 8, 3)

    # [N, 8, 3]
    corners = torch.matmul(rotation, corners.transpose(-2, -1)).transpose(-2, -1) + translation.unsqueeze(-2)

    return corners


def get_yaw_rotation(heading):
    """Gets rotation matrix for yaw rotation.

    Args:
        heading: [N] tensor of heading angles in radians.

    Returns:
        [N, 3, 3] rotation matrix.
    """
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    zeros = torch.zeros_like(cos_h)
    ones = torch.ones_like(cos_h)

    rotation = torch.stack([
            torch.stack([cos_h, -sin_h, zeros], dim=-1),
            torch.stack([sin_h, cos_h, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    return rotation
