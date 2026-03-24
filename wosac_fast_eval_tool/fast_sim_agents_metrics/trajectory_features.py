from typing import Tuple
import numpy as np
import torch


def central_diff(t: torch.Tensor, pad_value: float) -> torch.Tensor:
    """Computes the central difference along the last axis.

    This function is used to compute 1st order derivatives (speeds) when called
    once. Calling this function twice is used to compute 2nd order derivatives
    (accelerations) instead.
    This function returns the central difference as
    df(x)/dx = [f(x+h)-f(x-h)] / 2h.

    Args:
        t: A float Tensor of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        A Tensor of shape [..., steps] containing the central differences,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    # Prepare the tensor containing the value(s) to pad the result with.
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, device=t.device)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)


def central_logical_and(t: torch.Tensor, pad_value: bool) -> torch.Tensor:
    """Computes the central `logical_and` along the last axis.

    This function is used to compute the validity tensor for 1st and 2nd order
    derivatives using central difference, where element [i] is valid only if
    both elements [i-1] and [i+1] are valid.

    Args:
        t: A bool Tensor of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        A Tensor of shape [..., steps] containing the central `logical_and`,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    # Prepare the tensor containing the value(s) to pad the result with.
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, device=t.device)
    diff_t = torch.logical_and(t[..., 2:], t[..., :-2])
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)





def compute_kinematic_features(
        trajectories: torch.Tensor,
        seconds_per_step: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes kinematic features (speeds and accelerations).

    Note: Everything is assumed to be valid, filtering must be done afterwards.
    To maintain the original tensor length, speeds are prepended and appended
    with 1 np.nan, while accelerations with 2 np.nan (since central difference
    invalidated the two extremes).

    Args:
        trajectories: (num_rollouts, num_agents,num_steps,[x,y,z,heading])
        seconds_per_step: The duration (in seconds) of one step. This is used to
            scale speed and acceleration properly. This is always a positive value,
            usually defaulting to `submission_specs.STEP_DURATION_SECONDS`.

    Returns:
        A tuple containing the following 4 tensors:
            linear_speed: Magnitude of speed in (x, y, z). Shape (..., num_steps).
            linear_acceleration: Linear signed acceleration (changes in linear speed).
                Shape (..., num_steps).
            angular_speed: Angular speed (changes in heading). Shape (..., num_steps).
            angular_acceleration: Angular acceleration (changes in `angular_speed`).
                Shape (..., num_steps).
    """
    # Linear speed and acceleration.
    
    dpos = central_diff(trajectories.permute(3,0,1,2)[0:3], pad_value=np.nan)
    linear_speed = torch.norm(
            dpos, p=2, dim=0) / seconds_per_step
    linear_accel = central_diff(linear_speed, pad_value=np.nan) / seconds_per_step
    # Angular speed and acceleration. Since heading is normalized between
    # [-pi, pi] in Scenarios, we need to correctly wrap deltas between headings.
    # Given 2 headings, there are 2 possible deltas (one acute angle and one
    # obtuse), and by wrapping the delta in [-pi, pi] we are effectively selecting
    # the acute solution (the smaller angle). Using central difference, we are
    # estimating the delta from 2-steps, and we need to make sure the 2-steps
    # delta is acute, otherwise we could be actually selecting the wrong solution
    # for the 1-step delta. To do this, we scale the central difference by 2
    # before wrapping and then scale back. This effectively reduces the maximum
    # delta we can measure as pi/2 rad/step, which at 10Hz corresponds to
    # 5pi/sec, or 2.5 rotations on it own axis per second.
    dh_step = _wrap_angle(central_diff(trajectories[:,:,:,3], pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = _wrap_angle(central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step**2)
    return linear_speed, linear_accel, dh, d2h


def compute_kinematic_validity(valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return validity tensors for speeds and accelerations.

    Since we compute speed and acceleration directly from x/y/z/heading as
    central differences, we need to make sure to properly transform the validity
    tensors to match the new fields. The requirement is for both the steps used to
    compute the difference to be valid in order for the result to be valid.
    This is applied once for speeds and twice for accelerations, following the
    same strategy used to compute the kinematic fields.

    Args:
        valid: A boolean tensor of shape (..., num_steps) containing whether a
            certain object is valid at that step.

    Returns:
        speed_validity: A validity tensor that applies to speeds fields, where
            `central_logical_and` is applied once.
        acceleration_validity: A validity tensor that applies to acceleration
            fields, where `central_logical_and` is applied twice.
    """
    speed_validity = central_logical_and(valid, pad_value=False)
    acceleration_validity = central_logical_and(speed_validity, pad_value=False)
    return speed_validity, acceleration_validity


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wraps angles in the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
