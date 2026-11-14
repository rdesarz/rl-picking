# Environment design

## Control loop

Physics runs at 120 Hz and the policy runs at 30 Hz (`decimation = 4`). The six actions are relative position targets
for shoulder pan/lift, elbow flex, wrist flex/roll, and the gripper. Arm increments are limited to 0.033 rad per policy
step and the gripper to 0.02 rad. This matches a real position-servo interface and avoids asking a five-DOF arm to track
arbitrary six-DOF Cartesian poses.

## Scene

Each cloned environment contains the official Isaac Lab SO-101 USD, a 35 mm cube, and a 46 cm square table. The robot
uses the new-calibration joint convention. The fixed and moving fingertip offsets are taken into account when computing
the grasp center; the motor-link origin is not used as a proxy for the jaw center.

The wrist camera is the camera prim included in the official SO-101 asset. The optional external camera is a simulated
fixed, oblique RGB view. Both policy images are 64×64 at 30 Hz. The dual-camera tensor has six channels: wrist RGB
followed by external RGB.

## Markov decision process

The state teacher observes joint position, velocity and targets; the previous action; gripper and object geometry;
object velocity and height; and gripper closure. Camera actors observe only RGB, noisy encoder-space proprioception,
joint targets, and their previous action. Their critic sees the privileged state during training.

Reward terms are intentionally easy to inspect:

- approach the cube with the midpoint between the two fingertip pads;
- close the jaw near the cube and penalize closing far away;
- lift the cube, with a large bonus above 10 cm;
- lightly penalize action changes, high joint velocity, joint-limit pressure, and throwing.

Episodes end on a 10 cm lift, workspace exit/fall, abnormal robot state, or a 12 second timeout.

## First experiments

1. Run `zero_agent` and verify the cube stays on the table.
2. Run one visible `random_agent` environment and verify camera framing and jaw contacts.
3. Train state PPO and plot success rate, episode length, and each reward component.
4. Play the state checkpoint for at least 100 randomized episodes.
5. Distill to wrist RGB; only then compare direct visual PPO and dual-camera input.

If the state policy cannot exceed a useful success rate, do not tune the CNN. Inspect contact friction, robot/cube scale,
the home pose, action scale, and the fingertip/object distance first.

