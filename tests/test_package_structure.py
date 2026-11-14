"""Dependency-free checks that catch accidental task/config wiring regressions."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_all_registered_tasks_have_configs():
    registration = (ROOT / "src/so101_grasp/tasks/grasp/config/__init__.py").read_text()
    assert "SO101-Grasp-Cube-State-v0" in registration
    assert "SO101-Grasp-Cube-Wrist-v0" in registration
    assert "SO101-Grasp-Cube-DualCamera-v0" in registration
    assert "SO101GraspEnvCfg" in registration
    assert "SO101WristGraspEnvCfg" in registration
    assert "SO101DualCameraGraspEnvCfg" in registration


def test_joint_order_is_documented_and_stable():
    config = (ROOT / "src/so101_grasp/tasks/grasp/config/env_cfg.py").read_text()
    expected = '["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]'
    assert expected in config
    sim2real = (ROOT / "docs/SIM2REAL.md").read_text()
    for joint_name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
        assert joint_name in sim2real


def test_laptop_environment_defaults_are_conservative():
    state_config = (ROOT / "src/so101_grasp/tasks/grasp/config/env_cfg.py").read_text()
    camera_config = (ROOT / "src/so101_grasp/tasks/grasp/config/camera_env_cfg.py").read_text()
    assert "SO101GraspSceneCfg(num_envs=512" in state_config
    assert "SO101WristSceneCfg(num_envs=128" in camera_config
    assert "SO101DualCameraSceneCfg(num_envs=64" in camera_config


def test_windows_helpers_use_registered_task_id():
    setup_script = (ROOT / "scripts/windows/setup.ps1").read_text()
    assert "SO101-Grasp-Cube-State-v0" in setup_script
    assert "SO101-Grasp-State-v0" not in setup_script
    assert "uv sync --extra isaacsim" in setup_script


def test_isaac_lab_official_urllib3_override_is_applied_at_workspace_root():
    project = (ROOT / "pyproject.toml").read_text()
    assert "override-dependencies = [" in project
    assert '"urllib3>=2.8.0"' in project


def test_isaac_sim_opencv_is_resolved_from_nvidia_index():
    project = (ROOT / "pyproject.toml").read_text()
    assert '"opencv-python-headless-noffmpeg==4.14.0.94rc3"' in project
    assert 'opencv-python-headless-noffmpeg = { index = "nvidia" }' in project
