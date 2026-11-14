"""RSL-RL PPO configurations for the state and camera policies."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _distribution():
    return RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=0.20,
        std_type="log",
        std_range=(0.05, 0.35),
    )


@configclass
class SO101StatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 48
    max_iterations = 1200
    save_interval = 50
    experiment_name = "so101_cube_grasp_state"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_distribution(),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SO101WristPPORunnerCfg(SO101StatePPORunnerCfg):
    max_iterations = 3000
    save_interval = 100
    experiment_name = "so101_cube_grasp_wrist"
    obs_groups = {"actor": ["vision", "proprioception"], "critic": ["critic"]}
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="elu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(
            init_std=0.20,
            std_type="log",
            std_range=(0.05, 0.35),
        ),
    )


@configclass
class SO101DualCameraPPORunnerCfg(SO101WristPPORunnerCfg):
    experiment_name = "so101_cube_grasp_dual_camera"

