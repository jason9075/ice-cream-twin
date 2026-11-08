// The simulation runs in the Rust server (`icecream web`, rust/cli/src/web.rs); this page only draws it.
//
// Remote: the WebSocket. Requests carry an id and resolve with the reply; render frames (binary) are
// decoded as they arrive: grid patches go straight to the renderer's 3D textures, the rest (poses,
// particles, state) is kept as "latest" for the next animation frame.
//
// RemoteSim: what the panel and window.sim use, shaped like the old in-page simulation: params, a spoon
// with pos / quat / targetPos / targetQuat / setTarget, getState(), eat(), ...
import * as M from '../math.js';

const MAGIC = 0x31454349; // "ICE1"

/** Decode one binary frame (layout: rust/cli/src/web.rs). */
export function decodeFrame(buf) {
  const dv = new DataView(buf);
  let o = 0;
  const u32 = () => {
    const v = dv.getUint32(o, true);
    o += 4;
    return v;
  };
  if (u32() !== MAGIC) throw new Error('not an ICE1 frame');
  const seq = u32();
  const jl = u32();
  const state = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, o, jl)));
  o = (o + jl + 3) & ~3;
  const spoon = new Float32Array(buf, o, 7);
  const ghost = new Float32Array(buf, o + 28, 7);
  o += 56;
  const armOn = u32() !== 0;
  const links = new Float32Array(buf, o, 8 * 16);
  o += 8 * 16 * 4;
  const n = u32();
  const particles = { n, data: new DataView(buf, o, n * 16) };
  o += n * 16;
  let spilled = null;
  if (u32() === 1) {
    const ns = u32();
    spilled = { n: ns, data: new DataView(buf, o, ns * 16) };
    o += ns * 16;
  }
  const np = u32();
  const patches = [];
  for (let k = 0; k < np; k++) {
    const [grid, kind, x, y, z, sx, sy, sz] = [u32(), u32(), u32(), u32(), u32(), u32(), u32(), u32()];
    const bytes = sx * sy * sz * (kind === 0 ? 1 : 4);
    patches.push({ grid, kind, x, y, z, sx, sy, sz, data: new Uint8Array(buf, o, bytes) });
    o = (o + bytes + 3) & ~3;
  }
  if (o !== buf.byteLength) throw new Error(`frame length ${o} != ${buf.byteLength}`);
  return { seq, state, spoon, ghost, armOn, links, particles, spilled, patches };
}

export class Remote {
  constructor(url) {
    this.url = url;
    this.nextId = 1;
    this.pending = new Map();
    this.onHello = null;
    this.onFrame = null;
    this.onStatus = null;
    this.seq = 0;
    this.seqWaiters = [];
  }

  connect() {
    const ws = new WebSocket(this.url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => this.onStatus?.('connected');
    ws.onclose = () => {
      this.onStatus?.('disconnected');
      for (const p of this.pending.values()) p.reject(new Error('disconnected'));
      this.pending.clear();
      setTimeout(() => this.connect(), 1000); // the server may be restarting
    };
    ws.onmessage = (e) => {
      if (typeof e.data !== 'string') {
        const f = decodeFrame(e.data);
        this.seq = f.seq;
        this.onFrame?.(f);
        this.seqWaiters = this.seqWaiters.filter((w) => (f.seq >= w.seq ? (w.resolve(f), false) : true));
        return;
      }
      const m = JSON.parse(e.data);
      if (m.type === 'hello') {
        this.send({ type: 'config', stream: 'render' });
        this.onHello?.(m);
      } else if (m.id != null && this.pending.has(m.id)) {
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        if (m.type === 'error') p.reject(new Error(m.message));
        else p.resolve(m);
      } else if (m.type === 'error') {
        console.warn('simulation server:', m.message);
      }
    };
  }

  send(msg) {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  /** Send a command and resolve with its reply. */
  request(msg) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState !== 1) return reject(new Error('not connected to the simulation server'));
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ ...msg, id }));
    });
  }

  /** Resolve once a render frame at least this recent has arrived. */
  waitSeq(seq) {
    if (this.seq >= seq) return Promise.resolve();
    return new Promise((resolve) => this.seqWaiters.push({ seq, resolve }));
  }
}

/** The spoon as the panel sees it: the server's pose, and a target the page owns while it edits it. */
class RemoteSpoon {
  constructor(remote) {
    this.remote = remote;
    this.pos = new M.Vector3();
    this.quat = new M.Quaternion();
    this.targetPos = new M.Vector3();
    this.targetQuat = new M.Quaternion();
    this.edited = 0;
  }
  setTarget(pos, quat) {
    const msg = { type: 'set_target' };
    if (pos) {
      this.targetPos.copy(pos);
      msg.position = pos.toArray();
    }
    if (quat) {
      this.targetQuat.copy(quat).normalize();
      msg.quaternion = this.targetQuat.toArray();
    }
    this.edited = performance.now();
    this.remote.send(msg);
  }
  _update(state) {
    this.pos.set(...state.spoon.position);
    this.quat.set(...state.spoon.quaternion);
    // jogging and dragging build on the page's own target; otherwise follow the server's
    if (performance.now() - this.edited > 300) {
      this.targetPos.set(...state.target.position);
      this.targetQuat.set(...state.target.quaternion);
    }
  }
}

export class RemoteSim {
  constructor(remote, hello) {
    this.remote = remote;
    this.scene = hello.scene;
    // the jog frame is a page setting; everything else mirrors the server's parameters
    this.params = { ...hello.params, spoon: { ...hello.params.spoon, frame: 'world' } };
    this.spoon = new RemoteSpoon(remote);
    this.state = null;
    this.paused = hello.paused;
  }
  _frame(f) {
    this.state = f.state;
    this.paused = f.state.paused;
    this.spoon._update(f.state);
  }
  getState() {
    return this.state;
  }
  async eat() {
    return (await this.remote.request({ type: 'eat' })).grams;
  }
  releaseContent() {
    this.remote.send({ type: 'release' });
  }
  reset() {
    this.spoon.edited = 0;
    return this.remote.request({ type: 'reset' });
  }
  refill() {
    this.remote.send({ type: 'refill' });
  }
  home() {
    this.spoon.edited = 0;
    this.remote.send({ type: 'home' });
  }
  /** Merge parameter changes ({fluid: {viscosity: 0.5}}) on the server; resolves with all of them. */
  async setParams(patch) {
    const server = {};
    for (const [g, vals] of Object.entries(patch || {})) {
      const v = { ...vals };
      if (g === 'spoon' && 'frame' in v) {
        this.params.spoon.frame = v.frame;
        delete v.frame;
      }
      if (Object.keys(v).length) server[g] = v;
    }
    if (!Object.keys(server).length) return this.params;
    const r = await this.remote.request({ type: 'set_params', params: server });
    for (const [g, vals] of Object.entries(r.params)) Object.assign(this.params[g], vals);
    return this.params;
  }
}
