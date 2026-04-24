from __future__ import annotations

import torch


def _eps_like(x: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(torch.finfo(x.dtype).eps, dtype=x.dtype, device=x.device)


def make_quaternion(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.stack([w, x, y, z], dim=-1)


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


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(angle_axis, dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta <= _eps_like(theta)
    k = torch.where(small, 0.5 - theta * theta / 48.0, torch.sin(half) / theta)
    return normalize_quaternion(torch.cat([torch.cos(half), k * angle_axis], dim=-1))


def quaternion_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    sin_theta = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    cos_theta = q[..., :1]
    two_theta = 2.0 * torch.atan2(sin_theta, cos_theta)
    small = sin_theta <= _eps_like(sin_theta)
    k = torch.where(small, 2.0 / cos_theta.clamp_min(torch.finfo(q.dtype).tiny), two_theta / sin_theta)
    return k * q[..., 1:]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)
    two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
    row0 = torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1)
    row1 = torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)], dim=-1)
    row2 = torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    return quaternion_to_rotation_matrix(angle_axis_to_quaternion(angle_axis))


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=0.0))
    qx = 0.5 * torch.sign(m[..., 2, 1] - m[..., 1, 2]) * torch.sqrt(
        torch.clamp(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=0.0)
    )
    qy = 0.5 * torch.sign(m[..., 0, 2] - m[..., 2, 0]) * torch.sqrt(
        torch.clamp(1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=0.0)
    )
    qz = 0.5 * torch.sign(m[..., 1, 0] - m[..., 0, 1]) * torch.sqrt(
        torch.clamp(1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=0.0)
    )
    return normalize_quaternion(torch.stack([qw, qx, qy, qz], dim=-1))


def rotation_matrix_to_angle_axis(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(matrix))


def unit_quaternion_rotate_point(q: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    zeros = torch.zeros_like(point[..., :1])
    p = torch.cat([zeros, point], dim=-1)
    rotated = quaternion_product(quaternion_product(q, p), quaternion_conjugate(q))
    return rotated[..., 1:]


def quaternion_rotate_point(q: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    return unit_quaternion_rotate_point(normalize_quaternion(q), point)


def angle_axis_rotate_point(angle_axis: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    return unit_quaternion_rotate_point(angle_axis_to_quaternion(angle_axis), point)


def convert_ceres_to_eigen_quaternion(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def convert_eigen_to_ceres_quaternion(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)

