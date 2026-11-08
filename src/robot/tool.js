// End-of-arm tool modules: how the spoon is held by the SO-101.
//
// A module says where the spoon sits in the `gripper_link` frame and what happens to the gripper jaw.
// Today the stock gripper is clamped shut on the handle. The real arm will carry a different module
// (later a 3D-printed spoon holder): add an entry here with its own `spoonInGripper`, set `jaw: null`
// and `hideJaw: true`, and list its meshes. Nothing else in the code depends on the gripper.
import { SPOON } from '../sim/sdf.js';
import { m4 } from './so101.js';

/**
 * Spoon pose in gripper_link from a grip description: the point `gripX` along the handle's last
 * segment (spoon-local x, metres) is placed at `at` (gripper_link, metres), the handle runs along
 * gripper_link +Z (back toward the wrist) and the cup opening (spoon +Y) faces gripper_link +X
 * (the moving-jaw side).
 */
export function handleGrip(gripX, at) {
  const H = SPOON.handle;
  const a = H[H.length - 2];
  const b = H[H.length - 1];
  const t = (gripX - a[0]) / (b[0] - a[0]);
  const G = [gripX, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
  const al = Math.atan2(b[1] - a[1], b[0] - a[0]); // handle rises by this angle in spoon x-y
  const c = Math.cos(al);
  const s = Math.sin(al);
  // columns = images of spoon x, y, z in gripper_link
  //   spoon x -> c*Zg - s*Xg,  spoon y -> s*Zg + c*Xg,  spoon z -> Yg
  const R = [
    [-s, c, 0],
    [0, 0, 1],
    [c, s, 0],
  ];
  const M = m4.identity();
  for (let r = 0; r < 3; r++) {
    for (let k = 0; k < 3; k++) M[r * 4 + k] = R[r][k];
    M[r * 4 + 3] = at[r] - (R[r][0] * G[0] + R[r][1] * G[1] + R[r][2] * G[2]);
  }
  return M;
}

export const TOOLS = {
  /**
   * Stock SO-101 gripper clamped on the spoon handle. The handle lies against the fixed jaw's
   * fingertip pad (inner face at x = -7.9 mm in gripper_link), so its axis is at x = -7.9 + 3.2 mm.
   * The jaw angle is where the moving jaw first touches the 6.4 mm handle (found by sweeping the jaw
   * against the decimated meshes, see tools/so101_grip.mjs).
   */
  'so101-gripper-clamp': {
    label: 'SO-101 gripper, jaw clamped on the handle',
    jaw: -0.119,
    hideJaw: false,
    meshes: [],
    spoonInGripper: handleGrip(0.1, [-0.0079 + SPOON.handleR, 0, -0.09]),
  },
};

export const DEFAULT_TOOL = 'so101-gripper-clamp';
