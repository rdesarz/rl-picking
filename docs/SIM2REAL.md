# Sim-to-real checklist

The included environment is a starting point, not a turnkey hardware controller. A policy should not command the real
arm until the coordinate, timing, and safety checks below pass.

## 1. Measure your setup

- Calibrate the follower with LeRobot and preserve the calibration file used for deployment.
- Record the six calibrated joint readings at the simulated home pose.
- Measure cube size/mass, tabletop height, robot base pose, camera intrinsics, and camera-to-gripper transform.
- Mount every camera rigidly. Do not hand-hold or reposition it between training data and evaluation.
- Use even diffuse lighting and lock exposure/white balance if the camera supports it.

The SO-101 asset uses the new-calibration convention. Per-joint offsets between your encoders and simulation must be
measured; another robot's offsets are not safe defaults.

## 2. Match policy I/O

The actor expects observations at 30 Hz and outputs six normalized actions in this order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Arm actions map to a 0.033 rad relative target increment and the gripper action to 0.02 rad. Clamp resulting targets to
your calibrated hardware limits. Reproduce image crop, resize, RGB channel order, `[0, 1]` normalization, and camera
ordering exactly. Reject stale frames rather than silently repeating arbitrarily old images.

## 3. Improve randomization from measurements

The current ranges are deliberately modest. Log real step responses and grasp trials, then update:

- per-joint zero error, deadband, latency, speed, stiffness/damping, and backlash;
- jaw/object and table/object friction;
- camera mount error, focal length, principal point, exposure, blur, color, and end-to-end latency;
- object geometry, mass, center of mass, and spawn pose.

Randomization should cover plausible hardware behavior, not compensate for a wrong nominal model.

## 4. Stage hardware evaluation

1. Run inference with motor torque disabled and verify observation/action ordering.
2. Hold the arm clear of the table, apply a small action on one joint at a time, and verify direction and scale.
3. Enforce a low command-rate/step-size limit and an operator-held emergency stop.
4. Test reaching above an empty table before enabling gripper closure.
5. Test a light, compliant object inside a small workspace.
6. Increase speed only after reviewing joint targets, tracking error, frame age, and collisions.

Add an independent watchdog that stops on stale camera/servo data, communication loss, limit violation, excessive
tracking error, or workspace exit. A learned policy must never be the only safety layer.

## External-camera placement

Place a fixed RGB camera about 45 degrees above the table, looking across rather than straight down, so it sees the
cube and both jaws during closure. A top-down view localizes XY well but hides lift height; a low side view sees height
but suffers occlusion. The oblique compromise in the simulated dual-camera task is intended as a starting pose.

Use 640×480 at 30 fps on hardware and downsample in the deployment preprocessing. A standard UVC camera is enough for
the included RGB policy. A short-range depth camera such as the D405 is useful only if you add depth observations and
corresponding depth noise/dropout to simulation.

