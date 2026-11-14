# SO-101 grasping in Isaac Lab

This repository provides an Isaac Lab 3.0 training environment for grasping and lifting a randomized cube with an
SO-101 follower arm. It uses Isaac Lab's official calibrated SO-101 asset, relative joint-position commands at 30 Hz,
domain randomization, and three Gymnasium tasks:

| Task | Actor observations | Laptop default | Intended use |
|---|---|---:|---|
| `SO101-Grasp-Cube-State-v0` | Privileged robot/object state | 512 envs | Train the teacher and validate task physics |
| `SO101-Grasp-Cube-Wrist-v0` | Wrist RGB + proprioception | 128 envs | Deploy with your current hardware |
| `SO101-Grasp-Cube-DualCamera-v0` | Wrist + fixed external RGB + proprioception | 64 envs | Compare against a two-camera setup |

The camera tasks retain privileged state only for the training critic and teacher. It is not part of the deployed
actor input.

## Your Windows RTX 5090 laptop

The intended setup is native 64-bit Windows 11, not WSL. An RTX 5090 laptop GPU is suitable for this project. Isaac
Sim camera rendering also uses substantial system RAM and disk, so 32 GB RAM is the minimum and at least 100 GB of
free SSD space is recommended. The checked-in defaults are deliberately lower than workstation defaults to leave VRAM and thermal
headroom on a laptop.

Use a current NVIDIA notebook driver; 595.97 is NVIDIA's Windows test version for this Isaac Sim generation. Keep the laptop
plugged in, select its performance power profile, and force `python.exe`/Isaac Sim onto the high-performance NVIDIA
GPU in Windows Graphics settings if the machine has hybrid graphics.

Requirements:

- Windows 11 x86-64;
- Python 3.12 (managed automatically by `uv`);
- NVIDIA RTX GPU with at least 16 GB VRAM;
- 32 GB system RAM minimum;
- current [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

Isaac Lab is pinned to 3.0.0 because its APIs and quaternion convention differ from earlier releases.

## Windows install and smoke test

Open PowerShell in the repository root. Install `uv` if it is not already present:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell window after installation. Then allow local scripts for this window only and run the hardware
check:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\windows\check_system.ps1
```

If the check says Windows long paths are disabled, run this once in an **Administrator PowerShell**, then reopen the
normal terminal:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

Install the pinned dependencies and run a short headless simulation:

```powershell
.\scripts\windows\setup.ps1 -SmokeTest
```

The first Isaac Sim launch displays NVIDIA's license prompt. Read and accept it to continue. For a non-interactive
session, set `$env:OMNI_KIT_ACCEPT_EULA = "YES"` only after you have read and accepted those terms.

If an older checkout reports that Isaac Sim requests `urllib3==2.7.0` while Isaac Lab requests `urllib3>=2.8.0`, pull
the current `pyproject.toml` and re-resolve the environment:

```powershell
uv lock --upgrade-package urllib3
uv sync --extra isaacsim
uv run --extra isaacsim python -c "import urllib3; print(urllib3.__version__)"
```

The expected result is 2.8.0 or newer. Do not manually downgrade to 2.7.0: Isaac Lab 3.0 publishes 2.8.0 as its
tested override for the older Isaac Sim metadata.

Isaac Sim 6.2 also requires NVIDIA's custom `opencv-python-headless-noffmpeg==4.14.0.94rc3` build. The workspace routes
that package to NVIDIA's index explicitly. If an older lockfile cannot find it, refresh the package and verify the
installed distribution:

```powershell
uv lock --refresh-package opencv-python-headless-noffmpeg
uv sync --extra isaacsim
uv run --extra isaacsim python -c "from importlib.metadata import version; print(version('opencv-python-headless-noffmpeg'))"
```

Do not add `opencv-python` or `opencv-python-headless` alongside it: all three distributions install the same `cv2`
module and can overwrite one another.

Inspect one environment with a visible renderer before training:

```powershell
uv run --extra isaacsim isaaclab random_agent --task SO101-Grasp-Cube-State-v0 --num_envs 1 presets=isaacsim_physx,isaacsim_rtx
```

The initial dependency download is large and can take a while. If the visible window opens on the integrated GPU,
close it, change the Windows per-app graphics preference to **High performance**, and retry.

## Recommended training sequence

First train a state teacher. This separates contact/reward problems from vision problems and is substantially easier
to debug than starting with pixels. The task's 512-environment laptop default is used when `--num_envs` is omitted:

```powershell
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task SO101-Grasp-Cube-State-v0 --visualizer none presets=isaacsim_physx
```

Then distill that teacher into the wrist policy. Replace the checkpoint path with the state run you want to use:

```powershell
$teacher = (Resolve-Path ".\logs\rsl_rl\so101_grasp_state\RUN_NAME\model_XXXX.pt").Path
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task SO101-Grasp-Cube-Wrist-v0 --checkpoint "$teacher" --visualizer none presets=isaacsim_physx,isaacsim_rtx
```

To train wrist RGB directly with asymmetric actor-critic PPO instead:

```powershell
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task SO101-Grasp-Cube-Wrist-v0 --agent rsl_rl_ppo_cfg_entry_point --visualizer none presets=isaacsim_physx,isaacsim_rtx
```

Swap in `SO101-Grasp-Cube-DualCamera-v0` to test whether a fixed external view improves robustness. Start with the
checked-in defaults. Watch GPU memory and temperature in another PowerShell window with `nvidia-smi -l 2`; increase
`--num_envs` gradually only after a run is stable. If you see an out-of-memory error, halve the environment count.

Reasonable first overrides for this laptop are:

- state task: `--num_envs 512`, then try 1024;
- wrist camera: `--num_envs 128`, then try 192 or 256;
- dual camera: `--num_envs 64`, then try 96 or 128.

## Linux alternative

Linux remains supported for a separate workstation. The equivalent setup and smoke test are:

```bash
uv sync --extra isaacsim
uv run --extra isaacsim isaaclab zero_agent \
  --task SO101-Grasp-Cube-State-v0 --num_envs 16 --max_steps 200 \
  --visualizer none presets=isaacsim_physx
```

## What is randomized

- cube XY position and yaw;
- cube mass, friction, and restitution;
- robot reset pose;
- encoder zero bias and observation noise;
- RGB sensor noise.

The reward combines fingertip-to-object distance, closing near the object, lift height, and a success bonus. Closing
far from the object is penalized. Success is a 10 cm physical lift, not a proximity heuristic.

## Should you buy another camera?

Start with the wrist-only task; it is enough for a single object on a controlled tabletop. A wrist camera becomes
occluded by the jaw and object at the exact moment grasp quality matters, and its view changes quickly with arm
motion. A fixed external camera therefore usually helps when object poses, clutter, or recovery behavior matter.

For the cheapest useful upgrade, use a rigidly mounted UVC RGB webcam at 640x480/30 fps, roughly 45 degrees above the
table, with the whole workspace, gripper, and object visible. Match that physical pose by editing
[`camera_env_cfg.py`](src/so101_grasp/tasks/grasp/config/camera_env_cfg.py). If you specifically want depth, the
RealSense D405 is a better fit for this small workspace than long-range depth cameras. Depth is not yet part of these
policy observations, so do not buy it merely to run the included tasks.

Read [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for task details and [`docs/SIM2REAL.md`](docs/SIM2REAL.md) before
commanding the physical robot.

## Scope and validation status

The package and Windows helpers are statically checked in this workspace. Isaac Lab, Isaac Sim, an NVIDIA runtime,
and an SO-101 are not installed here, so simulator and hardware execution must still be validated on the target
laptop. Begin with the smoke test and inspect contact behavior before spending GPU time on training.
