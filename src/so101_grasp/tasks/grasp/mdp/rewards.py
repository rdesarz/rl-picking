"""Reward shaping for reaching, pinching, and lifting the object."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import gripper_closedness, gripper_to_object

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_object(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.08,
) -> torch.Tensor:
    """Smooth reward for moving the gripper link toward the object center."""
    distance = torch.linalg.vector_norm(gripper_to_object(env, gripper_cfg, object_cfg), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def pinch_near_object(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.045,
) -> torch.Tensor:
    """Reward closing only after the object is between the jaws."""
    distance = torch.linalg.vector_norm(gripper_to_object(env, gripper_cfg, object_cfg), dim=-1)
    near = torch.exp(-torch.square(distance / std))
    return near * gripper_closedness(env, gripper_joint_cfg).squeeze(-1)


def closing_far_from_object(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    minimum_distance: float = 0.06,
) -> torch.Tensor:
    """Penalize premature closure, a common local optimum in grasp training."""
    distance = torch.linalg.vector_norm(gripper_to_object(env, gripper_cfg, object_cfg), dim=-1)
    return (distance > minimum_distance).float() * gripper_closedness(env, gripper_joint_cfg).squeeze(-1)


def lift_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_top_z: float,
    target_height: float = 0.10,
) -> torch.Tensor:
    """Dense lift progress from tabletop to the target height."""
    height = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - table_top_z
    return (height / target_height).clamp(0.0, 1.0)


def lifted_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_top_z: float,
    minimum_height: float = 0.10,
) -> torch.Tensor:
    """Binary success reward for a physical lift."""
    height = env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - table_top_z
    return (height > minimum_height).float()


def object_velocity_l2(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Small regularizer against throwing the object."""
    velocity = env.scene[object_cfg.name].data.root_lin_vel_w.torch
    return torch.sum(torch.square(velocity), dim=-1)

