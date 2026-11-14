"""DAgger-style state-teacher to visual-student distillation."""

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


class BoundedTeacherDistillation(Distillation):
    """Clip teacher labels to the action range actually executed by the environment."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        return actions


def _student_model() -> RslRlCNNModelCfg:
    return RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="elu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.05, std_type="log"),
    )


@configclass
class SO101WristDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """The camera student acts while the frozen state teacher labels its states."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 1800
    save_interval = 100
    experiment_name = "so101_cube_grasp_wrist_distillation"
    run_name = ""
    obs_groups = {"student": ["vision", "proprioception"], "teacher": ["teacher_state"]}
    clip_actions = 1.0
    student = _student_model()
    # This must mirror SO101StatePPORunnerCfg.actor for checkpoint loading.
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.20,
            std_type="log",
            std_range=(0.05, 0.35),
        ),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=(
            "so101_grasp.tasks.grasp.config.agents.rsl_rl_distillation_cfg:BoundedTeacherDistillation"
        ),
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class SO101DualCameraDistillationRunnerCfg(SO101WristDistillationRunnerCfg):
    experiment_name = "so101_cube_grasp_dual_camera_distillation"

