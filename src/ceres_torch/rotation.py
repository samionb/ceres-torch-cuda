from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuaternionOrder:
    kW: int
    kX: int
    kY: int
    kZ: int


CeresQuaternionOrder = QuaternionOrder(0, 1, 2, 3)
EigenQuaternionOrder = QuaternionOrder(3, 0, 1, 2)


def _eps_like(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(torch.finfo(x.dtype).eps, dtype=x.dtype, device=x.device)


def _quaternion_order(order: QuaternionOrder | str | None) -> QuaternionOrder:
    if order is None or order is CeresQuaternionOrder or order == "ceres":
        return CeresQuaternionOrder
    if order is EigenQuaternionOrder or order == "eigen":
        return EigenQuaternionOrder
    if isinstance(order, QuaternionOrder):
        return order
    raise ValueError("order must be CeresQuaternionOrder, EigenQuaternionOrder, 'ceres', or 'eigen'")


def make_quaternion(
    w: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    order: QuaternionOrder | str | None = None,
) -> torch.Tensor:
    order = _quaternion_order(order)
    values = [None, None, None, None]
    values[order.kW] = w
    values[order.kX] = x
    values[order.kY] = y
    values[order.kZ] = z
    return torch.stack(values, dim=-1)  # type: ignore[arg-type]


def convert_quaternion(
    q: torch.Tensor,
    *,
    from_order: QuaternionOrder | str | None = None,
    to_order: QuaternionOrder | str | None = None,
) -> torch.Tensor:
    from_order = _quaternion_order(from_order)
    to_order = _quaternion_order(to_order)
    components = (q[..., from_order.kW], q[..., from_order.kX], q[..., from_order.kY], q[..., from_order.kZ])
    values = [None, None, None, None]
    values[to_order.kW] = components[0]
    values[to_order.kX] = components[1]
    values[to_order.kY] = components[2]
    values[to_order.kZ] = components[3]
    return torch.stack(values, dim=-1)  # type: ignore[arg-type]


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)


def quaternion_product(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    z0, z1, z2, z3 = z.unbind(-1)
    w0, w1, w2, w3 = w.unbind(-1)
    return torch.stack(
        [
            z0 * w0 - z1 * w1 - z2 * w2 - z3 * w3,
            z0 * w1 + z1 * w0 + z2 * w3 - z3 * w2,
            z0 * w2 - z1 * w3 + z2 * w0 + z3 * w1,
            z0 * w3 + z1 * w2 - z2 * w1 + z3 * w0,
        ],
        dim=-1,
    )


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    q_conj = quaternion_conjugate(q)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
    return q_conj / norm_sq


def cross_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(a, b, dim=-1)


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=-1)


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(angle_axis, dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta <= _eps_like(theta)
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    k = torch.where(small, 0.5 - theta * theta / 48.0, torch.sin(half) / safe_theta)
    return normalize_quaternion(torch.cat([torch.cos(half), k * angle_axis], dim=-1))


def quaternion_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    sin_theta = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    cos_theta = q[..., :1]
    small = sin_theta <= _eps_like(sin_theta)
    sign = torch.where(cos_theta < 0.0, -torch.ones_like(cos_theta), torch.ones_like(cos_theta))
    two_theta = 2.0 * torch.atan2(sign * sin_theta, sign * cos_theta)
    safe_sin = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    k = torch.where(small, 2.0 * torch.ones_like(sin_theta), two_theta / safe_sin)
    return k * q[..., 1:]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    return quaternion_to_scaled_rotation_matrix(q)


def quaternion_to_scaled_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    norm_sq = w * w + x * x + y * y + z * z
    row0 = torch.stack([norm_sq - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1)
    row1 = torch.stack([two * (x * y + z * w), norm_sq - two * (x * x + z * z), two * (y * z - x * w)], dim=-1)
    row2 = torch.stack([two * (x * z - y * w), two * (y * z + x * w), norm_sq - two * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    return quaternion_to_rotation_matrix(q)


def quaternion_to_scaled_rotation(q: torch.Tensor) -> torch.Tensor:
    return quaternion_to_scaled_rotation_matrix(q)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    return quaternion_to_rotation_matrix(angle_axis_to_quaternion(angle_axis))


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    eps = torch.finfo(m.dtype).eps
    positive_trace = trace >= 0.0

    t_trace = torch.sqrt(torch.clamp(trace + 1.0, min=eps))
    q_trace = torch.stack(
        [
            0.5 * t_trace,
            (m[..., 2, 1] - m[..., 1, 2]) * (0.5 / t_trace),
            (m[..., 0, 2] - m[..., 2, 0]) * (0.5 / t_trace),
            (m[..., 1, 0] - m[..., 0, 1]) * (0.5 / t_trace),
        ],
        dim=-1,
    )

    diag = torch.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], dim=-1)
    major = torch.argmax(diag, dim=-1)

    tx = torch.sqrt(torch.clamp(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=eps))
    qx = torch.stack(
        [
            (m[..., 2, 1] - m[..., 1, 2]) * (0.5 / tx),
            0.5 * tx,
            (m[..., 0, 1] + m[..., 1, 0]) * (0.5 / tx),
            (m[..., 0, 2] + m[..., 2, 0]) * (0.5 / tx),
        ],
        dim=-1,
    )
    ty = torch.sqrt(torch.clamp(1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=eps))
    qy = torch.stack(
        [
            (m[..., 0, 2] - m[..., 2, 0]) * (0.5 / ty),
            (m[..., 0, 1] + m[..., 1, 0]) * (0.5 / ty),
            0.5 * ty,
            (m[..., 1, 2] + m[..., 2, 1]) * (0.5 / ty),
        ],
        dim=-1,
    )
    tz = torch.sqrt(torch.clamp(1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=eps))
    qz = torch.stack(
        [
            (m[..., 1, 0] - m[..., 0, 1]) * (0.5 / tz),
            (m[..., 0, 2] + m[..., 2, 0]) * (0.5 / tz),
            (m[..., 1, 2] + m[..., 2, 1]) * (0.5 / tz),
            0.5 * tz,
        ],
        dim=-1,
    )

    q_negative = torch.where((major == 0).unsqueeze(-1), qx, torch.where((major == 1).unsqueeze(-1), qy, qz))
    return normalize_quaternion(torch.where(positive_trace.unsqueeze(-1), q_trace, q_negative))


def rotation_matrix_to_angle_axis(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(matrix))


def unit_quaternion_rotate_point(q: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    u = q[..., 1:]
    uv = 2.0 * cross_product(u, point)
    return point + q[..., :1] * uv + cross_product(u, uv)


def quaternion_rotate_point(q: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    return unit_quaternion_rotate_point(normalize_quaternion(q), point)


def angle_axis_rotate_point(angle_axis: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    return unit_quaternion_rotate_point(angle_axis_to_quaternion(angle_axis), point)


def rotation_matrix_rotate_point(matrix: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    return torch.matmul(matrix, point.unsqueeze(-1)).squeeze(-1)


def euler_angles_to_rotation_matrix(euler_degrees: torch.Tensor) -> torch.Tensor:
    """Ceres legacy Euler helper: X/Y/Z angles in degrees, applied as Rz Ry Rx."""

    radians = euler_degrees * (torch.pi / 180.0)
    return euler_angles_to_rotation(radians, axes="XYZ", intrinsic=False)


def rotation_matrix_to_euler_angles(matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`euler_angles_to_rotation_matrix` for the Ceres legacy convention."""

    m = matrix
    cy = torch.hypot(m[..., 0, 0], m[..., 1, 0])
    zero = torch.zeros_like(cy)
    regular_x = torch.atan2(m[..., 2, 1], m[..., 2, 2])
    regular_y = torch.atan2(-m[..., 2, 0], cy)
    regular_z = torch.atan2(m[..., 1, 0], m[..., 0, 0])
    singular_x = torch.atan2(-m[..., 1, 2], m[..., 1, 1])
    singular_y = torch.atan2(-m[..., 2, 0], cy)
    singular = cy <= _eps_like(cy)
    angles = torch.stack(
        [
            torch.where(singular, singular_x, regular_x),
            torch.where(singular, singular_y, regular_y),
            torch.where(singular, zero, regular_z),
        ],
        dim=-1,
    )
    return angles * (180.0 / torch.pi)


def euler_angles_to_rotation(
    euler_radians: torch.Tensor,
    *,
    axes: str = "XYZ",
    intrinsic: bool = False,
) -> torch.Tensor:
    if len(axes) != 3 or any(axis.upper() not in {"X", "Y", "Z"} for axis in axes):
        raise ValueError("axes must be a three-character string containing only X, Y, and Z")
    angles = euler_radians.reshape(*euler_radians.shape[:-1], 3)
    matrices = [_axis_rotation_matrix(angles[..., i], axis.upper()) for i, axis in enumerate(axes)]
    result = torch.eye(3, dtype=euler_radians.dtype, device=euler_radians.device).expand(angles.shape[:-1] + (3, 3))
    order = matrices if intrinsic else list(reversed(matrices))
    for matrix in order:
        result = result @ matrix
    return result


def _axis_rotation_matrix(angle: torch.Tensor, axis: str) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        rows = [
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, c, -s], dim=-1),
            torch.stack([zero, s, c], dim=-1),
        ]
    elif axis == "Y":
        rows = [
            torch.stack([c, zero, s], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-s, zero, c], dim=-1),
        ]
    else:
        rows = [
            torch.stack([c, -s, zero], dim=-1),
            torch.stack([s, c, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ]
    return torch.stack(rows, dim=-2)


def convert_ceres_to_eigen_quaternion(q: torch.Tensor) -> torch.Tensor:
    return convert_quaternion(q, from_order=CeresQuaternionOrder, to_order=EigenQuaternionOrder)


def convert_eigen_to_ceres_quaternion(q: torch.Tensor) -> torch.Tensor:
    return convert_quaternion(q, from_order=EigenQuaternionOrder, to_order=CeresQuaternionOrder)


AngleAxisToQuaternion = angle_axis_to_quaternion
QuaternionToAngleAxis = quaternion_to_angle_axis
RotationMatrixToQuaternion = rotation_matrix_to_quaternion
RotationMatrixToAngleAxis = rotation_matrix_to_angle_axis
AngleAxisToRotationMatrix = angle_axis_to_rotation_matrix
EulerAnglesToRotationMatrix = euler_angles_to_rotation_matrix
EulerAnglesToRotation = euler_angles_to_rotation
RotationMatrixToEulerAngles = rotation_matrix_to_euler_angles
QuaternionToScaledRotation = quaternion_to_scaled_rotation
QuaternionToRotation = quaternion_to_rotation
UnitQuaternionRotatePoint = unit_quaternion_rotate_point
QuaternionRotatePoint = quaternion_rotate_point
QuaternionProduct = quaternion_product
QuaternionConjugate = quaternion_conjugate
CrossProduct = cross_product
DotProduct = dot_product
AngleAxisRotatePoint = angle_axis_rotate_point
MakeQuaternion = make_quaternion
ConvertQuaternion = convert_quaternion
ConvertCeresToEigenQuaternion = convert_ceres_to_eigen_quaternion
ConvertEigenToCeresQuaternion = convert_eigen_to_ceres_quaternion
