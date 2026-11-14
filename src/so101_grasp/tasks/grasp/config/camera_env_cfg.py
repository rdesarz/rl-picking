"""Wrist-only and wrist-plus-external visual SO-101 grasp tasks."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from so101_grasp.tasks.grasp import mdp

from .env_cfg import JOINTS, SO101GraspEnvCfg, SO101GraspSceneCfg, StatePolicyCfg

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


def _renderer_cfg() -> MultiBackendRendererCfg:
    cfg = MultiBackendRendererCfg()
    cfg.default.enable_shadows = True
    cfg.ovrtx.enable_shadows = True
    return cfg


@configclass
class SO101WristSceneCfg(SO101GraspSceneCfg):
    """Base scene with the camera already modeled in the official SO-101 asset."""

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera/pinhole",
        spawn=PinholeCameraCfg(focal_length=13.6, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(convention="opengl"),
        data_types=["rgb"],
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=_renderer_cfg(),
        background_color=(0.25, 0.25, 0.25),
    )


@configclass
class SO101DualCameraSceneCfg(SO101WristSceneCfg):
    """Wrist camera plus a fixed oblique overview camera."""

    external_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ExternalCamera",
        spawn=PinholeCameraCfg(focal_length=18.0, horizontal_aperture=24.0, clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.45, -0.45, 0.40),
            rot=(-0.2597851, 0.1589718, 0.8124452, 0.4971644),
            convention="world",
        ),
        data_types=["rgb"],
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=_renderer_cfg(),
        background_color=(0.25, 0.25, 0.25),
    )


@configclass
class WristVisionCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.camera_rgb,
        params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        clip=(0.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class DualVisionCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.dual_camera_rgb,
        params={
            "wrist_cfg": SceneEntityCfg("wrist_camera"),
            "external_cfg": SceneEntityCfg("external_camera"),
        },
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        clip=(0.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class ProprioceptionCfg(ObsGroup):
    """Only quantities available on the physical robot at inference time."""

    joint_pos = ObsTerm(
        func=mdp.encoder_joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    joint_target = ObsTerm(
        func=mdp.joint_target,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)},
        noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005),
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class WristObservationsCfg:
    vision: WristVisionCfg = WristVisionCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: StatePolicyCfg = StatePolicyCfg()
    critic: StatePolicyCfg = StatePolicyCfg()


@configclass
class DualCameraObservationsCfg:
    vision: DualVisionCfg = DualVisionCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: StatePolicyCfg = StatePolicyCfg()
    critic: StatePolicyCfg = StatePolicyCfg()


@configclass
class SO101WristGraspEnvCfg(SO101GraspEnvCfg):
    """Deployable wrist-camera actor with a privileged critic during training."""

    # Camera tensors dominate VRAM use; start small on a laptop GPU.
    scene: SO101WristSceneCfg = SO101WristSceneCfg(num_envs=128, env_spacing=0.75, replicate_physics=True)
    observations: WristObservationsCfg = WristObservationsCfg()

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)


@configclass
class SO101DualCameraGraspEnvCfg(SO101GraspEnvCfg):
    """Deployable six-channel actor using wrist and fixed external RGB."""

    scene: SO101DualCameraSceneCfg = SO101DualCameraSceneCfg(num_envs=64, env_spacing=0.75, replicate_physics=True)
    observations: DualCameraObservationsCfg = DualCameraObservationsCfg()

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
