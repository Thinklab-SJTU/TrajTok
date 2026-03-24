import math
import numpy as np
import torch
from . import trajectory_features

# Constant distance to apply when distances between objects are invalid. This
# will avoid the propagation of nans and should be reduced out when taking the
# minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Collision threshold, i.e. largest distance between objects that is considered
# to be a collision.
COLLISION_DISTANCE_THRESHOLD = 0.0
# Rounding factor to apply to the corners of the object boxes in distance and
# collision computation. The rounding factor is between 0 and 1, where 0 yields
# rectangles with sharp corners (no rounding) and 1 yields capsule shapes.
# Default value of 0.7 conservately fits most vehicle contours.
CORNER_ROUNDING_FACTOR = 0.7

# Condition thresholds for filtering obstacles driving ahead of the ego pbject
# when computing the time-to-collision metric. This metric only considers
# collisions in lane-following a.k.a. tailgating situations.
# Maximum allowed difference in heading.
MAX_HEADING_DIFF = math.radians(75.0)  # radians.
# Maximum allowed difference in heading in case of small lateral overlap.
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)  # radians.
# Lateral overlap threshold below which the tighter heading alignment condition
# `_MAX_HEADING_DIFF_FOR_SMALL_OVERLAP` is used.
SMALL_OVERLAP_THRESHOLD = 0.5  # meters.

# Maximum time-to-collision, in seconds, used to clip large values or in place
# of invalid values.
MAXIMUM_TIME_TO_COLLISION = 5.0

NUM_VERTICES_IN_BOX = 4

def _get_object_following_mask(
        longitudinal_distance: torch.Tensor,
        lateral_overlap: torch.Tensor,
        yaw_diff: torch.Tensor,
) -> torch.Tensor:
    """Returns a mask for objects that are being followed.

    Args:
        longitudinal_distance: A float tensor of shape (num_evaluated_objects,
            num_objects, num_steps) containing the longitudinal distance between all
            pairs of objects.
        lateral_overlap: A float tensor of shape (num_evaluated_objects,
            num_objects, num_steps) containing the lateral overlap between all pairs
            of objects.
        yaw_diff: A float tensor of shape (num_evaluated_objects, num_objects,
            num_steps) containing the heading difference between all pairs of objects.

    Returns:
        A boolean tensor of shape (num_evaluated_objects, num_objects, num_steps)
        indicating whether each object is being followed by each evaluated object.
    """
    # Objects are being followed if they are ahead of the evaluated object
    # (positive longitudinal distance) and have a small lateral overlap.
    # Shape: (num_evaluated_objects, num_objects, num_steps).

    valid_mask = longitudinal_distance > 0.0
    valid_mask = torch.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)
    valid_mask = torch.logical_and(valid_mask, lateral_overlap < 0.0)



    # Combine all conditions.
    return torch.logical_and(
            valid_mask,
            torch.logical_or(
                    lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
                    yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
            ),
    )

def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the signed magnitude of cross product of 2d vectors.

    Args:
        a: A tensor with shape (..., 2).
        b: A tensor with the same shape as `a`.

    Returns:
        An (n-1)-rank tensor that stores the cross products of paired 2d vectors in
        `a` and `b`.
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _get_downmost_edge_in_box(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Note: We assume box edges are given in a counter-clockwise order, so that
    the edge which starts with the downmost vertex (i.e. the downmost edge) is
    uniquely identified.

    Args:
        box: (num_boxes, num_points_per_box, 2). The last dimension contains the x-y
            coordinates of corners in boxes.

    Returns:
        A tuple of two tensors:
            downmost_vertex_idx: The index of the downmost vertex, which is also the
                index of the downmost edge. Shape: (num_boxes, 1).
            downmost_edge_direction: The tangent unit vector of the downmost edge,
                pointing in the counter-clockwise direction of the box.
                Shape: (num_boxes, 1, 2).
    """
    # The downmost vertex is the lowest in the y dimension.
    # Shape: (num_boxes, 1).
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1).unsqueeze(-1)

    # Find the counter-clockwise point edge from the downmost vertex.
    edge_start_vertex = torch.gather(box, 1, downmost_vertex_idx.unsqueeze(-1).expand(-1, -1, 2))
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = torch.gather(box, 1, edge_end_idx.unsqueeze(-1).expand(-1, -1, 2))

    # Compute the direction of this downmost edge.
    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(-1)
    return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(
        polygon_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Tensor containing the vertices of each polygon, with
            shape (num_polygons, num_points_per_polygon, 2). Each polygon is assumed
            to have an equal number of vertices.

    Returns:
        tangent_unit_vectors: A unit vector in (x,y) with the same direction as
            the tangent to the edge. Shape: (num_polygons, num_points_per_polygon, 2).
        normal_unit_vectors: A unit vector in (x,y) with the same direction as
            the normal to the edge.
            Shape: (num_polygons, num_points_per_polygon, 2).
        edge_lengths: Lengths of the edges.
            Shape (num_polygons, num_points_per_polygon).
    """
    # Shift the polygon points by 1 position to get the edges.
    # Shape: (num_polygons, 1, 2).
    first_point_in_polygon = polygon_points[:, 0:1, :]
    # Shape: (num_polygons, num_points_per_polygon, 2).
    shifted_polygon_points = torch.cat(
            [polygon_points[:, 1:, :], first_point_in_polygon], dim=1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    edge_vectors = shifted_polygon_points - polygon_points

    # Shape: (num_polygons, num_points_per_polygon).
    edge_lengths = torch.norm(edge_vectors, dim=-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    # Shape: (num_polygons, num_points_per_polygon, 2).
    normal_unit_vectors = torch.stack(
            [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1)
    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def minkowski_sum_of_box_and_box_points(box1_points: torch.Tensor,
                                                                            box2_points: torch.Tensor) -> torch.Tensor:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    The last dimensions of the input and return store the x and y coordinates of
    the points. Both box1_points and box2_points needs to be stored in
    counter-clockwise order. Otherwise the function will return incorrect results
    silently.

    Args:
        box1_points: Tensor of vertices for box 1, with shape:
            (num_boxes, num_points_per_box, 2).
        box2_points: Tensor of vertices for box 2, with shape:
            (num_boxes, num_points_per_box, 2).

    Returns:
        The Minkowski sum of the two boxes, of size (num_boxes,
        num_points_per_box * 2, 2). The points will be stored in counter-clockwise
        order.
    """
    # Hard coded order to pick points from the two boxes. This is a simplification
    # of the generic convex polygons case. For boxes, the adjacent edges are
    # always 90 degrees apart from each other, so the index of vertices can be
    # hard coded.
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=box1_points.device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64, device=box1_points.device)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(
            box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(
            box2_points)

    # The cross-product of the unit vectors indicates whether the downmost edge
    # in box2 is pointing to the left side (the inward side of the resulting
    # Minkowski sum) of the downmost edge in box1. If this is the case, pick
    # points from box1 in the order `point_order_2`, and pick points from box2 in
    # the order of `point_order_1`. Otherwise, we switch the order to pick points
    # from the two boxes, pick points from box1 in the order of `point_order_1`,
    # and pick points from box2 in the order of `point_order_2`.
    # Shape: (num_boxes, 1)
    condition = (
            cross_product_2d(
                    downmost_box1_edge_direction, downmost_box2_edge_direction
            )
            >= 0.0
    )
    # Tile condition to shape: (num_boxes, num_points_per_box * 2 = 8).
    condition = condition.expand(-1, 8)

    # box1_point_order of size [num_boxes, num_points_per_box * 2 = 8].
    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    # Shift box1_point_order by box1_start_idx, so that the first index in
    # box1_point_order is the downmost vertex in the box.
    box1_point_order = torch.remainder(box1_point_order + box1_start_idx,
                                                                     NUM_VERTICES_IN_BOX)
    # Gather points from box1 in order.
    # ordered_box1_points is of size [num_boxes, num_points_per_box * 2, 2].
    ordered_box1_points = torch.gather(
            box1_points, 1, box1_point_order.unsqueeze(-1).expand(-1, -1, 2))

    # Gather points from box2 as well.
    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = torch.remainder(box2_point_order + box2_start_idx,
                                                                     NUM_VERTICES_IN_BOX)
    ordered_box2_points = torch.gather(
            box2_points, 1, box2_point_order.unsqueeze(-1).expand(-1, -1, 2))
    minkowski_sum = ordered_box1_points + ordered_box2_points
    return minkowski_sum


def signed_distance_from_point_to_convex_polygon(
        query_points: torch.Tensor, polygon_points: torch.Tensor) -> torch.Tensor:
    """Finds the signed distances from query points to convex polygons.

    Each polygon is represented by a 2d tensor storing the coordinates of its
    vertices. The vertices must be ordered in counter-clockwise order. An
    arbitrary number of pairs (point, polygon) can be batched on the 1st
    dimension.

    Note: Each polygon is associated to a single query point.

    Args:
        query_points: (batch_size, 2). The last dimension is the x and y
            coordinates of points.
        polygon_points: (batch_size, num_points_per_polygon, 2). The last
            dimension is the x and y coordinates of vertices.

    Returns:
        A tensor containing the signed distances of the query points to the
        polygons. Shape: (batch_size,).
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = (
            _get_edge_info(polygon_points))

    # Expand the shape of `query_points` to (num_polygons, 1, 2), so that
    # it matches the dimension of `polygons_points` for broadcasting.
    query_points = query_points.unsqueeze(1)
    # Compute query points to polygon points distances.
    # Shape (num_polygons, num_points_per_polygon, 2).
    vertices_to_query_vectors = query_points - polygon_points
    # Shape (num_polygons, num_points_per_polygon).
    vertices_distances = torch.norm(vertices_to_query_vectors, dim=-1)

    # Query point to edge distances are measured as the perpendicular distance
    # of the point from the edge. If the projection of this point on to the edge
    # falls outside the edge itself, this distance is not considered (as there)
    # will be a lower distance with the vertices of this specific edge.

    # Make distances negative if the query point is in the inward side of the
    # edge. Shape: (num_polygons, num_points_per_polygon).
    edge_signed_perp_distances = torch.sum(
            -normal_unit_vectors * vertices_to_query_vectors, dim=-1)

    # If `edge_signed_perp_distances` are all less than 0 for a
    # polygon-query_point pair, then the query point is inside the convex polygon.
    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=-1)

    # Project the distances over the tangents of the edge, and verify where the
    # projections fall on the edge.
    # Shape: (num_polygons, num_edges_per_polygon).
    projection_along_tangent = torch.sum(
            tangent_unit_vectors * vertices_to_query_vectors, dim=-1)
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths
    # Shape: (num_polygons, num_edges_per_polygon).
    is_projection_on_edge = torch.logical_and(
            projection_along_tangent_proportion >= 0.0,
            projection_along_tangent_proportion <= 1.0)

    # If the point projection doesn't lay on the edge, set the distance to inf.
    edge_perp_distances = torch.abs(edge_signed_perp_distances)
    edge_distances = torch.where(is_projection_on_edge,
                                                         edge_perp_distances, torch.tensor(float('inf'), device=edge_perp_distances.device))

    # Aggregate vertex and edge distances.
    # Shape: (num_polyons, 2 * num_edges_per_polygon).
    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances],
                                                                         dim=-1)
    # Aggregate distances per polygon and change the sign if the point lays inside
    # the polygon. Shape: (num_polygons,).
    min_distance = torch.min(edge_and_vertex_distance, dim=-1)[0]
    signed_distances = torch.where(is_inside, -min_distance, min_distance)
    return signed_distances

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


def get_transform(rotation, translation):
    """Gets 4x4 transform matrix from rotation and translation.

    Args:
        rotation: [..., 3, 3] rotation matrix.
        translation: [..., 3] translation vector.

    Returns:
        [..., 4, 4] transform matrix.
    """
    transform = torch.zeros(*rotation.shape[:-2], 4, 4, device=rotation.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


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

def compute_distance_to_nearest_object(
        boxes: torch.Tensor,
        valid: torch.Tensor,
        evaluated_object_mask: torch.Tensor,
        corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> torch.Tensor:
    """Computes the distance to nearest object for each of the evaluated objects.

    Objects are represented by 2D rectangles with rounded corners.

    Args:
        boxes: A float Tensor of shape (num_rollouts, num_objects, num_steps, [x,y,z,l,w,h,heading]) 
        valid: A boolean Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            validity of the objects over time.
        evaluated_object_mask: A boolean tensor of shape (num_rollouts, num_objects), to index the
            objects identified by the tensors defined above. If True, the object is
            considered part of the "evaluation set", i.e. the object can actively
            collide into other objects. If False, the object can also be passively
            collided into.
        corner_rounding_factor: Rounding factor to apply to the corners of the
            object boxes, between 0 (no rounding) and 1 (capsule shape rounding).

    Returns:
        A tensor of shape (num_rollouts, num_evaluated_objects, num_steps), containing the
        distance to the nearest object, for each timestep and for all the objects
        to be evaluated, as specified by `evaluated_object_mask`.
    """
    # Concatenate tensors to have the same convention as `box_utils`.
    num_rollouts, num_objects, num_steps, num_features = boxes.shape

    # Shrink the bounding boxes to get their rectangular "core". The rounded
    # rectangles we want to process are distance isolines of the rectangle cores.

    # The shrinking distance is half of the minimal dimension between length and
    # width, multiplied by the rounding factor.
    # Shape: [num_rollouts, num_objects, num_steps]
    shrinking_distance = (
            torch.minimum(boxes[..., 3], boxes[..., 4]) * corner_rounding_factor / 2.
    )
    # Box cores to use in distance computation below, after shrinking all sides
    # uniformly.
    boxes = torch.cat(
            [
                    boxes[..., :3],
                    boxes[..., 3:4] - 2.*shrinking_distance.unsqueeze(-1),
                    boxes[..., 4:5] - 2.*shrinking_distance.unsqueeze(-1),
                    boxes[..., 5:],
            ],
            dim=-1,
    )

    # Reshape for box_utils processing
    boxes = boxes.reshape(-1, num_features)  # Flatten all dimensions except features

    # Compute box corners using `box_utils`, and take xy coordinates of the lower
    # 4 corners (lower in terms of z-coordinate), as we are only computing
    # distances for 2D boxes.
    box_corners = get_upright_3d_box_corners(boxes)[:, :4, :2]
    box_corners = box_corners.reshape(num_rollouts, num_objects, num_steps, 4, 2)

    # Rearrange the boxes based on `evaluated_object_mask`. We want two sets of
    # boxes: the first one including just the evaluated objects, the second one
    # with all the boxes, but having the evaluated objects as first (this is used
    # later to filter out self distances).
    # `eval_corners` shape: (num_rollouts, num_evaluated_objects, num_steps, 4, 2).
    eval_corners = box_corners[:,evaluated_object_mask]
    num_eval_objects = eval_corners.shape[1]
    # `other_corners` shape: (num_rollouts, num_objects-num_evaluated_objects, num_steps, 4, 2).
    other_corners = box_corners[:,~evaluated_object_mask]
    # `all_corners` shape: (num_rollouts, num_objects, num_steps, 4, 2).
    all_corners = torch.cat([eval_corners, other_corners], dim=1)
    
    # Broadcast both sets for pair-wise comparisons.
    eval_corners = eval_corners.unsqueeze(2).expand(num_rollouts, num_eval_objects, num_objects, num_steps, 4, 2)
    all_corners = all_corners.unsqueeze(1).expand(num_rollouts, num_eval_objects, num_objects, num_steps, 4, 2)
    
    # Flatten the dimensions for processing
    eval_corners = eval_corners.reshape(-1, 4, 2)
    all_corners = all_corners.reshape(-1, 4, 2)
    
    # The signed distance between two polygons A and B is equal to the distance
    # between the origin and the Minkowski sum A + (-B), where we generate -B by a
    # reflection.
    neg_all_corners = -1.0 * all_corners
    minkowski_sum = minkowski_sum_of_box_and_box_points(
            box1_points=eval_corners, box2_points=neg_all_corners
    )
    
    # Shape: (num_rollouts * num_evaluated_objects * num_objects * num_steps, 8, 2).
    signed_distances_flat = (
            signed_distance_from_point_to_convex_polygon(
                    query_points=torch.zeros_like(minkowski_sum[:, 0, :]),
                    polygon_points=minkowski_sum,
            )
    )

    # Reshape back to original dimensions
    signed_distances = signed_distances_flat.reshape(num_rollouts, num_eval_objects, num_objects, num_steps)

    # Gather the shrinking distances for the evaluated objects and for all objects
    # after reordering.
    eval_shrinking_distance = shrinking_distance[:,evaluated_object_mask]
    other_shrinking_distance = shrinking_distance[:,~evaluated_object_mask]
    all_shrinking_distance = torch.cat(
            [eval_shrinking_distance, other_shrinking_distance], dim=1
    )

    # Recover distances between rounded boxes from the distances between core
    # boxes by subtracting the shrinking distances.
    signed_distances -= eval_shrinking_distance.unsqueeze(2)
    signed_distances -= all_shrinking_distance.unsqueeze(1)

    # Create self-mask for each rollout
    self_mask = torch.eye(num_eval_objects, num_objects, dtype=torch.float32, device=signed_distances.device)[
            None, :, :, None
    ].expand(num_rollouts, -1, -1, num_steps)
    signed_distances = signed_distances + self_mask * EXTREMELY_LARGE_DISTANCE

    # Mask out invalid boxes
    eval_validity = valid[evaluated_object_mask]
    other_validity = valid[~evaluated_object_mask]
    all_validity = torch.cat([eval_validity, other_validity], dim=0)
    valid_mask = torch.logical_and(eval_validity.unsqueeze(1), all_validity.unsqueeze(0)).squeeze(0).repeat(num_rollouts, 1, 1, 1)
    signed_distances = torch.where(
            valid_mask, signed_distances, EXTREMELY_LARGE_DISTANCE
    )
    
    # Aggregate over the "all objects" dimension.
    return torch.min(signed_distances, dim=2)[0]


def compute_time_to_collision_with_object_in_front(
        *,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        length: torch.Tensor,
        width: torch.Tensor,
        heading: torch.Tensor,
        valid: torch.Tensor,
        evaluated_object_mask: torch.Tensor,
        seconds_per_step: float,
) -> torch.Tensor:
    """Computes the time-to-collision of the evaluated objects.

    Args:
        center_x: A float Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            x-component of the object positions.
        center_y: A float Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            y-component of the object positions.
        length: A float Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            object lengths.
        width: A float Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            object widths.
        heading: torch.Tensor of shape (num_rollouts, num_objects, num_steps) containing the
            object headings, in radians.
        valid: A boolean Tensor of shape (num_objects, num_steps) containing the
            validity of the objects over time.
        evaluated_object_mask: A boolean tensor of shape (num_objects), to index the
            objects identified by the tensors defined above. If True, the object is
            considered part of the "evaluation set", i.e. the object can actively
            collide into other objects. If False, the object can also be passively
            collided into.
        seconds_per_step: The duration (in seconds) of one step. This is used to
            scale speed and acceleration properly. This is always a positive value,
            usually defaulting to `submission_specs.STEP_DURATION_SECONDS`.

    Returns:
        A tensor of shape (num_rollouts, num_evaluated_objects, num_steps), containing the
        time-to-collision, for each timestep and for all the objects to be
        evaluated, as specified by `evaluated_object_mask`.
    """
    num_rollouts, num_objects, num_steps = center_x.shape

    
    # Shape: (num_rollouts, num_objects, num_steps)
    speed = trajectory_features.compute_kinematic_features(
            trajectories=torch.stack([center_x, center_y, torch.zeros_like(center_x), heading], dim=-1),
            seconds_per_step=seconds_per_step,
    )[0][...,11:]

    
    # Shape: (num_rollouts, num_steps, num_objects, 6)
    boxes = torch.stack([
            center_x[...,11:].transpose(1, 2),  
            center_y[...,11:].transpose(1, 2),
            length.transpose(1, 2),
            width.transpose(1, 2),
            heading[...,11:].transpose(1, 2),
            speed.transpose(1, 2)
    ], dim=-1)
    valid = valid.transpose(0, 1)  # (num_steps, num_objects)

    eval_boxes = boxes[:, :, evaluated_object_mask]
    ego_xy = eval_boxes[..., :2]
    ego_sizes = eval_boxes[..., 2:4]
    ego_yaw = eval_boxes[..., 4:5]
    ego_speed = eval_boxes[..., 5:6]
    other_xy = boxes[..., :2]
    other_sizes = boxes[..., 2:4]
    other_yaw = boxes[..., 4:5]
    
    yaw_diff = torch.abs(other_yaw.unsqueeze(2) - ego_yaw.unsqueeze(3))

    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    other_long_offset = torch.sum(
            other_sizes.unsqueeze(2) / 2.0 * torch.abs(torch.cat([yaw_diff_cos, yaw_diff_sin], dim=-1)),
            dim=-1
    )
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    other_lat_offset = torch.sum(
            other_sizes.unsqueeze(2) / 2.0 * torch.abs(torch.cat([yaw_diff_sin, yaw_diff_cos], dim=-1)),
            dim=-1
    )

    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects, 2)
    other_relative_xy = rotate_2d_points(
            (other_xy.unsqueeze(2) - ego_xy.unsqueeze(3)),
            -ego_yaw.unsqueeze(3)
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    long_distance = (
            other_relative_xy[..., 0]
            - ego_sizes[..., 0:1] / 2.0
            - other_long_offset
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    lat_overlap = (
            torch.abs(other_relative_xy[..., 1])
            - ego_sizes[..., 1:2] / 2.0
            - other_lat_offset
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    following_mask = _get_object_following_mask(
            longitudinal_distance=long_distance,
            lateral_overlap=lat_overlap,
            yaw_diff=yaw_diff.squeeze(-1)
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    valid_mask = torch.logical_and(
            valid.unsqueeze(0).unsqueeze(2),
            following_mask
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects, num_objects)
    masked_long_distance = (
            long_distance
            + (1.0 - valid_mask.float()) * EXTREMELY_LARGE_DISTANCE
    )

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects)
    box_ahead_index = torch.argmin(masked_long_distance, dim=-1)

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects)
    distance_to_box_ahead = torch.gather(
            masked_long_distance, -1, box_ahead_index.unsqueeze(-1)
    ).squeeze(-1)
    # Shape: (num_rollouts, num_steps, num_evaluated_objects)
    box_ahead_speed = torch.gather(
            speed.transpose(1, 2).unsqueeze(2).expand(-1, -1, num_objects, -1),
            -1, box_ahead_index.unsqueeze(-1)
    ).squeeze(-1)

    
    # Shape: (num_rollouts, num_steps, num_evaluated_objects)
    rel_speed = ego_speed.squeeze(-1) - box_ahead_speed
    # Shape: (num_rollouts, num_steps, num_evaluated_objects)
    time_to_collision = torch.where(
            rel_speed > 0.0,
            torch.minimum(distance_to_box_ahead / rel_speed, torch.tensor(MAXIMUM_TIME_TO_COLLISION)),
            MAXIMUM_TIME_TO_COLLISION
    )

    
    # Shape: (num_rollouts, num_evaluated_objects, num_steps)
    return time_to_collision.transpose(1, 2)

def rotate_2d_points(xys: torch.Tensor, rotation_yaws: torch.Tensor) -> torch.Tensor:
    """Rotates `xys` counter-clockwise using the `rotation_yaws`.

    Rotates about the origin counter-clockwise in the x-y plane.

    Arguments may have differing shapes as long as they are broadcastable to a
    common shape.

    Args:
        xys: A float Tensor with shape (..., 2) containing xy coordinates.
        rotation_yaws: A float Tensor with shape (..., 1) containing angles in
            radians.

    Returns:
        A float Tensor with shape (..., 2) containing the rotated `xys`.
    """

    rotation_yaws = rotation_yaws.expand_as(xys[..., 0:1])
    
    rel_cos_yaws = torch.cos(rotation_yaws)
    rel_sin_yaws = torch.sin(rotation_yaws)

    xs_out = rel_cos_yaws * xys[..., 0:1] - rel_sin_yaws * xys[..., 1:2]
    ys_out = rel_sin_yaws * xys[..., 0:1] + rel_cos_yaws * xys[..., 1:2]
    
    return torch.cat([xs_out, ys_out], dim=-1)



