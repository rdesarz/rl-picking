"""Privileged-state SO-101 cube grasp-and-lift environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots import so101

from so101_grasp.tasks.grasp import mdp

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]
GRIPPER_BODIES = ["gripper", "moving_jaw_so101_v1"]

TABLE_TOP_Z = 0.035
OBJECT_SIZE = 0.035
OBJECT_START_Z = TABLE_TOP_Z + 0.5 * OBJECT_SIZE + 0.001

ROBOT_CFG = so101.SO101_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=so101.SO101_CFG.init_state.replace(
        pos=(-0.05, 0.0, 0.0),
        # Face the +X tabletop workspace. Isaac Lab 3 quaternions are XYZW.
        rot=(0.0, 0.0, 0.7071068, 0.7071068),
    ),
    soft_joint_pos_limit_factor=0.98,
)


@configclass
class SO101GraspSceneCfg(InteractiveSceneCfg):
    """One arm, a randomized cube, and a compact tabletop per environment."""

    robot = ROBOT_CFG

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.35, 0.95), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.23, 0.0, OBJECT_START_Z)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.46, 0.46, TABLE_TOP_Z),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.15, 0.12), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, 0.5 * TABLE_TOP_Z)),
    )

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=700.0, color=(1.0, 0.94, 0.86), angle=12.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3250576, 0.3250576, 0.0, 0.8880738)),
    )


@configclass
class ActionsCfg:
    """Relative servo targets at 30 Hz, matching hardware deployment semantics."""

    arm = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        scale=0.033,
        use_zero_offset=True,
    )
    gripper = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        preserve_order=True,
        scale=0.02,
        use_zero_offset=True,
    )


def _robot_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)


def _gripper_bodies() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=GRIPPER_BODIES, preserve_order=True)


@configclass
class StatePolicyCfg(ObsGroup):
    """Fully observed teacher policy inputs."""

    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": _robot_joints()})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": _robot_joints()})
    joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": _robot_joints()})
    previous_action = ObsTerm(func=mdp.last_action)
    gripper_position = ObsTerm(
        func=mdp.gripper_position_b,
        params={"gripper_cfg": _gripper_bodies(), "robot_cfg": SceneEntityCfg("robot")},
    )
    object_position = ObsTerm(func=mdp.object_position_b)
    gripper_to_object = ObsTerm(func=mdp.gripper_to_object, params={"gripper_cfg": _gripper_bodies()})
    object_motion = ObsTerm(func=mdp.object_motion)
    object_height = ObsTerm(func=mdp.object_height, params={"table_top_z": TABLE_TOP_Z})
    gripper_closedness = ObsTerm(
        func=mdp.gripper_closedness,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"])},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: StatePolicyCfg = StatePolicyCfg()
    critic: StatePolicyCfg = StatePolicyCfg()


@configclass
class EventsCfg:
    """Reset variation plus modest physics and encoder randomization."""

    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.45, 1.10),
            "dynamic_friction_range": (0.35, 1.00),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.02, 0.05),
            "operation": "abs",
        },
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_with_noise,
        mode="reset",
        params={"asset_cfg": _robot_joints(), "position_noise": 0.025},
    )
    reset_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "x_range": (0.18, 0.28),
            "y_range": (-0.09, 0.09),
        },
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "bound": 0.01},
    )


@configclass
class RewardsCfg:
    reach = RewTerm(
        func=mdp.reach_object,
        weight=2.0,
        params={"gripper_cfg": _gripper_bodies(), "object_cfg": SceneEntityCfg("object"), "std": 0.08},
    )
    pinch = RewTerm(
        func=mdp.pinch_near_object,
        weight=4.0,
        params={
            "gripper_cfg": _gripper_bodies(),
            "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    premature_close = RewTerm(
        func=mdp.closing_far_from_object,
        weight=-1.0,
        params={
            "gripper_cfg": _gripper_bodies(),
            "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    lift = RewTerm(
        func=mdp.lift_object,
        weight=20.0,
        params={"object_cfg": SceneEntityCfg("object"), "table_top_z": TABLE_TOP_Z, "target_height": 0.10},
    )
    success = RewTerm(
        func=mdp.lifted_success,
        weight=100.0,
        params={"object_cfg": SceneEntityCfg("object"), "table_top_z": TABLE_TOP_Z, "minimum_height": 0.10},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": _robot_joints()})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _robot_joints()})
    object_speed = RewTerm(
        func=mdp.object_velocity_l2,
        weight=-0.001,
        params={"object_cfg": SceneEntityCfg("object")},
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(
        func=mdp.object_lifted,
        params={"object_cfg": SceneEntityCfg("object"), "table_top_z": TABLE_TOP_Z, "minimum_height": 0.10},
    )
    object_lost = DoneTerm(
        func=mdp.object_out_of_workspace,
        params={"object_cfg": SceneEntityCfg("object")},
    )
    unstable_robot = DoneTerm(func=mdp.abnormal_robot_state)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    """PhysX through either the Isaac Sim or kit-less OV backend."""

    physx = PhysxAutoCfg(
        isaacsim_physx=PhysxCfg(bounce_threshold_velocity=0.01),
        ovphysx=OvPhysxCfg(enable_external_forces_every_iteration=False),
    )
    default = physx


@configclass
class SO101GraspEnvCfg(ManagerBasedRLEnvCfg):
    """State-teacher environment for randomized cube grasping."""

    # Conservative default for an RTX 5090 laptop. Increase with --num_envs
    # after the smoke tests and thermal behavior are stable.
    scene: SO101GraspSceneCfg = SO101GraspSceneCfg(num_envs=512, env_spacing=0.75, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.62, -0.48, 0.38), lookat=(0.21, 0.0, 0.06))

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
