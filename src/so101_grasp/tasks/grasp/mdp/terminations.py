"""Success and failure conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_top_z: float,
    minimum_height: float = 0.10,
) -> torch.Tensor:
    """Terminate when the cube reaches the target lift height."""
    return env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] > table_top_z + minimum_height


def object_out_of_workspace(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    xy_limit: float = 0.55,
    minimum_z: float = -0.05,
) -> torch.Tensor:
    """Terminate cubes that fall or leave their cloned workspace."""
    position = env.scene[object_cfg.name].data.root_pos_w.torch - env.scene.env_origins
    return (position[:, :2].abs() > xy_limit).any(dim=-1) | (position[:, 2] < minimum_z)


def abnormal_robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    maximum_joint_speed: float = 30.0,
) -> torch.Tensor:
    """Catch non-finite simulation state or an obviously unstable articulation."""
    robot = env.scene[asset_cfg.name]
    position = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    velocity = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    invalid = ~torch.isfinite(position).all(dim=-1) | ~torch.isfinite(velocity).all(dim=-1)
    return invalid | (velocity.abs() > maximum_joint_speed).any(dim=-1)
