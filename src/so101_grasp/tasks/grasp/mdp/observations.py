"""Deployable and privileged observations for grasp-and-lift."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)


def _first_id(ids: list[int] | slice) -> int:
    """Return the one ID selected by a single-body/single-joint SceneEntityCfg."""
    if isinstance(ids, slice):
        if ids.start is None:
            return 0
        return ids.start
    return ids[0]


def joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current position targets, which are available on the real servo interface."""
    return env.scene[asset_cfg.name].data.joint_pos_target.torch[:, asset_cfg.joint_ids]


def encoder_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions with an episode-constant calibration bias."""
    joint_pos = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    bias = getattr(env, "_so101_encoder_bias", 0.0)
    if isinstance(bias, torch.Tensor):
        bias = bias[:, asset_cfg.joint_ids]
    return joint_pos + bias


def object_position_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object position expressed in the robot base frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    position, _ = subtract_frame_transforms(
        robot.data.root_link_pos_w.torch,
        robot.data.root_link_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return position


def gripper_position_b(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gripper-link position expressed in the robot base frame."""
    robot = env.scene[robot_cfg.name]
    center = grasp_center_w(env, gripper_cfg)
    return quat_apply_inverse(robot.data.root_link_quat_w.torch, center - robot.data.root_link_pos_w.torch)


def grasp_center_w(env: ManagerBasedRLEnv, gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Midpoint between the SO-101's fixed and moving fingertip pads."""
    robot = env.scene[gripper_cfg.name]
    body_ids = gripper_cfg.body_ids
    if isinstance(body_ids, slice) or len(body_ids) != 2:
        raise ValueError("gripper_cfg must select [gripper, moving_jaw_so101_v1] in that order")
    pads = []
    for body_id, offset in zip(body_ids, (FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET), strict=True):
        position = robot.data.body_link_pos_w.torch[:, body_id]
        orientation = robot.data.body_link_quat_w.torch[:, body_id]
        local_offset = position.new_tensor(offset).expand_as(position)
        pads.append(position + quat_apply(orientation, local_offset))
    return 0.5 * (pads[0] + pads[1])


def gripper_to_object(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """World-frame vector from the gripper link to the object."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch - grasp_center_w(env, gripper_cfg)


def object_motion(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object linear and angular velocity in world coordinates."""
    obj = env.scene[object_cfg.name]
    return torch.cat((obj.data.root_lin_vel_w.torch, obj.data.root_ang_vel_w.torch), dim=-1)


def object_height(
    env: ManagerBasedRLEnv,
    table_top_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object center height above the tabletop."""
    return (env.scene[object_cfg.name].data.root_pos_w.torch[:, 2] - table_top_z).unsqueeze(-1)


def gripper_closedness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalized jaw closure: zero is open and one is closed."""
    robot = env.scene[asset_cfg.name]
    joint_id = _first_id(asset_cfg.joint_ids)
    position = robot.data.joint_pos.torch[:, joint_id]
    limits = robot.data.soft_joint_pos_limits.torch[:, joint_id]
    closedness = (limits[:, 1] - position) / (limits[:, 1] - limits[:, 0]).clamp_min(1.0e-6)
    return closedness.clamp(0.0, 1.0).unsqueeze(-1)


def camera_rgb(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normalized RGB in channel-first format for RSL-RL's CNN encoder."""
    image = env.scene.sensors[sensor_cfg.name].data.output["rgb"].torch[..., :3]
    image = image.float() / 255.0 if image.dtype == torch.uint8 else image.float().clamp(0.0, 1.0)
    return image.permute(0, 3, 1, 2).contiguous()


def dual_camera_rgb(
    env: ManagerBasedRLEnv,
    wrist_cfg: SceneEntityCfg,
    external_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Wrist and external RGB stacked into a six-channel image."""
    return torch.cat((camera_rgb(env, wrist_cfg), camera_rgb(env, external_cfg)), dim=1)
