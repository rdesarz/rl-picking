"""Reset and calibration-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def reset_robot_with_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    position_noise: float = 0.025,
) -> None:
    """Reset near the calibrated home pose and align the drive targets."""
    ids = _env_ids(env, env_ids)
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    position = robot.data.default_joint_pos.torch[ids][:, joint_ids].clone()
    position += torch.empty_like(position).uniform_(-position_noise, position_noise)
    limits = robot.data.soft_joint_pos_limits.torch[ids][:, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    velocity = torch.zeros_like(position)
    robot.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=ids)
    robot.set_joint_position_target_index(target=position, joint_ids=joint_ids, env_ids=ids)
    robot.set_joint_velocity_target_index(target=velocity, joint_ids=joint_ids, env_ids=ids)


def reset_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
) -> None:
    """Reset the object on the table with randomized XY position and yaw."""
    ids = _env_ids(env, env_ids)
    count = len(ids)
    obj = env.scene[asset_cfg.name]
    pose = obj.data.default_root_pose.torch[ids].clone()
    pose[:, 0] = torch.empty(count, device=env.device).uniform_(*x_range)
    pose[:, 1] = torch.empty(count, device=env.device).uniform_(*y_range)
    yaw = torch.empty(count, device=env.device).uniform_(*yaw_range)
    pose[:, 3:7] = 0.0
    pose[:, 5] = torch.sin(0.5 * yaw)
    pose[:, 6] = torch.cos(0.5 * yaw)
    pose[:, :3] += env.scene.env_origins[ids]
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=env.device), env_ids=ids)


def randomize_encoder_bias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bound: float = 0.01,
) -> None:
    """Sample per-joint zero-offset error visible to a deployable policy."""
    ids = _env_ids(env, env_ids)
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_so101_encoder_bias"):
        env._so101_encoder_bias = torch.zeros_like(robot.data.joint_pos.torch)
    env._so101_encoder_bias[ids] = torch.empty_like(env._so101_encoder_bias[ids]).uniform_(-bound, bound)

