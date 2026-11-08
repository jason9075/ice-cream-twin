import * as M from './math.js';

/**
 * Digital-twin interface.
 *
 *  - `window.sim` : JS API for scripts / Playwright / a parent page.
 *  - WebSocket client: connects to an external controller (e.g. a ROS node) that sends
 *    pose commands and receives state at a fixed rate.
 *
 * Frames: the scene is three.js Y-up (metres). Pass `frame: "ros"` to use ROS REP-103
 * Z-up instead (x forward, y left, z up); conversion is a +90° rotation about X.
 */

const Q_YUP_TO_ROS = new M.Quaternion().setFromAxisAngle(new M.Vector3(1, 0, 0), Math.PI / 2);
const Q_ROS_TO_YUP = Q_YUP_TO_ROS.clone().invert();

export const frames = {
  posToRos: (p) => [p[0], -p[2], p[1]],
  posFromRos: (p) => [p[0], p[2], -p[1]],
  quatToRos: (q) => Q_YUP_TO_ROS.clone().multiply(new M.Quaternion(...q)).multiply(Q_ROS_TO_YUP).toArray(),
  quatFromRos: (q) => Q_ROS_TO_YUP.clone().multiply(new M.Quaternion(...q)).multiply(Q_YUP_TO_ROS).toArray(),
};

function stateIn(state, frame) {
  if (frame !== 'ros') return state;
  const conv = (o) => ({ ...o, position: frames.posToRos(o.position), quaternion: frames.quatToRos(o.quaternion) });
  return {
    ...state,
    frame: 'ros',
    spoon: {
      ...conv(state.spoon),
      linearVelocity: frames.posToRos(state.spoon.linearVelocity),
      angularVelocity: frames.posToRos(state.spoon.angularVelocity),
    },
    target: conv(state.target),
    // the base yaw is about the up axis in both frames, so only the position changes
    ...(state.arm && {
      arm: {
        ...state.arm,
        goal: conv(state.arm.goal),
        base: { ...state.arm.base, position: frames.posToRos(state.arm.base.position) },
      },
    }),
  };
}

export function installApi(sim, hooks) {
  const api = {
    /** Set the spoon target pose. position [x,y,z] m, quaternion [x,y,z,w]. */
    setTarget({ position, quaternion, frame = 'yup' } = {}) {
      const p = position && (frame === 'ros' ? frames.posFromRos(position) : position);
      const q = quaternion && (frame === 'ros' ? frames.quatFromRos(quaternion) : quaternion);
      sim.spoon.setTarget(p ? new M.Vector3(...p) : null, q ? new M.Quaternion(...q) : null);
    },
    getState(frame = 'yup') {
      const st = sim.getState();
      if (hooks.arm) st.arm = hooks.arm.state();
      return stateIn(st, frame);
    },
    /**
     * Joint-space command for the SO-101 (URDF joint convention): shoulder_pan, shoulder_lift,
     * elbow_flex, wrist_flex, wrist_roll. The spoon target becomes the pose these angles give.
     */
    setJoints({ joints, unit = 'rad' } = {}) {
      if (!hooks.arm || !joints) return;
      const k = unit === 'deg' ? Math.PI / 180 : 1;
      hooks.arm.setJoints(sim, joints.map((v) => v * k));
    },
    /** Empty the spoon; a Promise of the grams eaten (the GPU simulation reads them back). */
    eat: () => hooks.eat(),
    release: () => sim.releaseContent(),
    reset: () => hooks.reset(),
    refill: () => sim.refill(),
    /** Deep-merge parameter overrides, e.g. {fluid: {viscosity: 0.5}}. */
    setParams(patch) {
      for (const [group, vals] of Object.entries(patch || {})) {
        if (sim.params[group]) Object.assign(sim.params[group], vals);
      }
      hooks.paramsChanged?.();
    },
    getParams: () => JSON.parse(JSON.stringify(sim.params)),
    /**
     * Run the simulation without rendering (e.g. for fast scripted tests). Returns a Promise of the
     * state: with the GPU simulation every frame waits for the previous one's statistics.
     */
    async stepFrames(n = 1, dt = 1 / 60) {
      for (let i = 0; i < n; i++) await (hooks.step ?? ((t) => sim.stepAsync(t)))(dt);
      return api.getState();
    },
    /** Which simulation runs: 'gpu' (compute shaders) or 'cpu' (reference); setting it reloads the page. */
    setSimulation(mode) {
      if (hooks.simulation) hooks.simulation.mode = mode;
      return hooks.simulation?.mode;
    },
    /** Pause the real-time loop (stepFrames still works). */
    pause: (v = true) => hooks.pause(v),
    frames,
    sim,
  };
  window.sim = api;
  return api;
}

export class WsBridge {
  constructor(api) {
    this.api = api;
    this.ws = null;
    this.status = 'disconnected';
    this.rateHz = 30;
    this.frame = 'yup';
    this._timer = null;
    this.onStatus = () => {};
  }

  connect(url) {
    this.disconnect();
    this._set('connecting');
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      this._set(`error: ${e.message}`);
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this._set('connected');
      ws.send(JSON.stringify({ type: 'hello', params: this.api.getParams() }));
      this._timer = setInterval(() => this.sendState(), 1000 / this.rateHz);
    };
    ws.onclose = () => {
      clearInterval(this._timer);
      if (this.ws === ws) this._set('disconnected');
    };
    ws.onerror = () => this._set('error (is the server running?)');
    ws.onmessage = (ev) => this.handle(ev.data);
  }

  disconnect() {
    clearInterval(this._timer);
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.close();
    }
    this._set('disconnected');
  }

  sendState() {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'state', ...this.api.getState(this.frame) }));
  }

  handle(raw) {
    let m;
    try {
      m = JSON.parse(raw);
    } catch {
      return;
    }
    const frame = m.frame || this.frame;
    switch (m.type) {
      case 'set_target':
        this.api.setTarget({ position: m.position, quaternion: m.quaternion, frame });
        break;
      case 'set_joints':
        this.api.setJoints({ joints: m.joints, unit: m.unit });
        break;
      case 'eat':
        this.api.eat();
        break;
      case 'release':
        this.api.release();
        break;
      case 'reset':
        this.api.reset();
        break;
      case 'refill':
        this.api.refill();
        break;
      case 'set_params':
        this.api.setParams(m.params);
        break;
      case 'get_state':
        this.sendState();
        break;
      case 'config':
        if (m.frame) this.frame = m.frame;
        if (m.rateHz) {
          this.rateHz = Math.max(1, Math.min(120, m.rateHz));
          clearInterval(this._timer);
          this._timer = setInterval(() => this.sendState(), 1000 / this.rateHz);
        }
        break;
    }
  }

  _set(s) {
    this.status = s;
    this.onStatus(s);
  }
}
