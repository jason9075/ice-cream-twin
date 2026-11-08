import * as M from '../math.js';
import { HOME_POS, HOME_QUAT } from '../sim/spoon.js';

/** Jog axes: [label, key, kind, axis, sign] */
const JOG = [
  ['X−', 'a', 'lin', 0, -1],
  ['X+', 'd', 'lin', 0, 1],
  ['Y+', 'r', 'lin', 1, 1],
  ['Y−', 'f', 'lin', 1, -1],
  ['Z−', 'w', 'lin', 2, -1],
  ['Z+', 's', 'lin', 2, 1],
  ['Rx−', 'u', 'ang', 0, -1],
  ['Rx+', 'o', 'ang', 0, 1],
  ['Ry−', 'j', 'ang', 1, -1],
  ['Ry+', 'l', 'ang', 1, 1],
  ['Rz−', 'k', 'ang', 2, -1],
  ['Rz+', 'i', 'ang', 2, 1],
];

const LEAD_LIN = 0.012; // m the target may run ahead of the actual spoon while jogging
const LEAD_ANG = M.MathUtils.degToRad(20);

export class Panel {
  constructor(sim, actions) {
    this.sim = sim;
    this.actions = actions;
    this.active = new Map(); // jog id -> true
    this.el = document.getElementById('panel');
    this.el.innerHTML = this.template();
    this.$ = (s) => this.el.querySelector(s);
    this.bind();
  }

  template() {
    const btn = (j, i) =>
      `<button class="jog" data-jog="${i}" title="key ${j[1].toUpperCase()}">${j[0]}<small>${j[1].toUpperCase()}</small></button>`;
    return `
<h1>🍨 Ice Cream Twin</h1>
<section>
  <div class="row">
    <button id="eat" class="primary">Eat 😋</button>
    <button id="release" title="tip the scoop off">Tip off</button>
    <button id="home">Home</button>
  </div>
</section>
<section>
  <h2>Jog <span class="seg" id="frame"><button data-f="world">World</button><button data-f="tool">Tool</button></span></h2>
  <div class="jogs">${JOG.slice(0, 6).map((j, i) => btn(j, i)).join('')}</div>
  <div class="jogs">${JOG.slice(6).map((j, i) => btn(j, i + 6)).join('')}</div>
  <label class="slider">Lin <input id="jl" type="range" min="0.005" max="0.15" step="0.005"><span id="jlv"></span></label>
  <label class="slider">Ang <input id="ja" type="range" min="5" max="180" step="5"><span id="jav"></span></label>
  <div class="hint">Mouse: <b>Ctrl+drag</b> moves the target on its horizontal plane, <b>Ctrl+wheel</b> changes its height.</div>
</section>
<section>
  <h2>Go to pose <small>(mm / deg, XYZ Euler)</small></h2>
  <div class="pose">
    ${['x', 'y', 'z', 'rx', 'ry', 'rz'].map((k) => `<label>${k}<input id="g_${k}" type="number" step="1"></label>`).join('')}
  </div>
  <div class="row"><button id="goto">Go</button><button id="fill">Use current</button></div>
</section>
<section>
  <h2>Status</h2>
  <div class="chips"><span id="c_contact">contact</span><span id="c_blocked">blocked</span><span id="c_cut">cutting</span></div>
  <table class="readout">
    <tr><th></th><th>x</th><th>y</th><th>z</th><th>rx</th><th>ry</th><th>rz</th></tr>
    <tr><td>actual</td>${'<td></td>'.repeat(6)}</tr>
    <tr><td>target</td>${'<td></td>'.repeat(6)}</tr>
  </table>
  <div class="kv" id="kv"></div>
</section>
<section class="help">
  Hold a jog button or key. <b>A/D</b> X · <b>R/F</b> Y · <b>W/S</b> Z · <b>U/O</b> Rx · <b>J/L</b> Ry · <b>K/I</b> Rz · <b>E</b> eat · <b>H</b> home.<br>
  The spoon follows the target with speed limits and cannot pass through bowls or the table; ice cream slows it by the estimated cutting force.
</section>`;
  }

  bind() {
    const sp = this.sim.params.spoon;
    this.el.querySelectorAll('.jog').forEach((b) => {
      const id = +b.dataset.jog;
      const on = (e) => {
        e.preventDefault();
        b.setPointerCapture?.(e.pointerId);
        this.active.set(id, true);
        b.classList.add('on');
      };
      const off = () => {
        this.active.delete(id);
        b.classList.remove('on');
      };
      b.addEventListener('pointerdown', on);
      b.addEventListener('pointerup', off);
      b.addEventListener('pointercancel', off);
      b.addEventListener('lostpointercapture', off);
    });
    const keyIdx = new Map(JOG.map((j, i) => [j[1], i]));
    const typing = (e) => /INPUT|TEXTAREA|SELECT/.test(e.target.tagName);
    window.addEventListener('keydown', (e) => {
      if (typing(e) || e.ctrlKey || e.metaKey) return;
      const k = e.key.toLowerCase();
      if (keyIdx.has(k)) {
        this.active.set(keyIdx.get(k), true);
        this.el.querySelector(`[data-jog="${keyIdx.get(k)}"]`)?.classList.add('on');
      } else if (k === 'e' && !e.repeat) this.actions.eat();
      else if (k === 'h' && !e.repeat) this.home();
    });
    window.addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      if (keyIdx.has(k)) {
        this.active.delete(keyIdx.get(k));
        this.el.querySelector(`[data-jog="${keyIdx.get(k)}"]`)?.classList.remove('on');
      }
    });
    window.addEventListener('blur', () => {
      this.active.clear();
      this.el.querySelectorAll('.jog.on').forEach((b) => b.classList.remove('on'));
    });

    const seg = (sel, attr, get, set) => {
      const root = this.$(sel);
      const refresh = () => root.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset[attr] === get()));
      root.querySelectorAll('button').forEach((b) =>
        b.addEventListener('click', () => {
          set(b.dataset[attr]);
          refresh();
        }),
      );
      refresh();
      return refresh;
    };
    this.refreshFrame = seg('#frame', 'f', () => sp.frame, (v) => (sp.frame = v));

    const slider = (id, key, fmt) => {
      const inp = this.$(`#${id}`);
      const out = this.$(`#${id}v`);
      inp.value = sp[key];
      out.textContent = fmt(sp[key]);
      inp.addEventListener('input', () => {
        sp[key] = +inp.value;
        out.textContent = fmt(sp[key]);
      });
    };
    slider('jl', 'jogLinSpeed', (v) => `${(v * 1000).toFixed(0)} mm/s`);
    slider('ja', 'jogAngSpeed', (v) => `${v}°/s`);

    this.$('#eat').addEventListener('click', () => this.actions.eat());
    this.$('#release').addEventListener('click', () => this.actions.release());
    this.$('#home').addEventListener('click', () => this.home());
    this.$('#goto').addEventListener('click', () => this.gotoPose());
    this.$('#fill').addEventListener('click', () => this.fillPose());
    this.fillPose();
  }

  home() {
    this.sim.spoon.setTarget(HOME_POS, HOME_QUAT);
  }

  fillPose() {
    const sp = this.sim.spoon;
    const e = new M.Euler().setFromQuaternion(sp.quat, 'XYZ');
    const v = [sp.pos.x * 1000, sp.pos.y * 1000, sp.pos.z * 1000, ...[e.x, e.y, e.z].map(M.MathUtils.radToDeg)];
    ['x', 'y', 'z', 'rx', 'ry', 'rz'].forEach((k, i) => (this.$(`#g_${k}`).value = v[i].toFixed(0)));
  }

  gotoPose() {
    const g = (k) => +this.$(`#g_${k}`).value || 0;
    const pos = new M.Vector3(g('x'), g('y'), g('z')).multiplyScalar(0.001);
    const d = M.MathUtils.degToRad;
    const q = new M.Quaternion().setFromEuler(new M.Euler(d(g('rx')), d(g('ry')), d(g('rz')), 'XYZ'));
    this.sim.spoon.setTarget(pos, q);
  }

  /** Integrate held jog buttons into the target pose. */
  applyJog(dt) {
    if (!this.active.size) return false;
    const sp = this.sim.spoon;
    const p = this.sim.params.spoon;
    const lin = new M.Vector3();
    const ang = new M.Vector3();
    for (const id of this.active.keys()) {
      const j = JOG[id];
      (j[2] === 'lin' ? lin : ang).setComponent(j[3], (j[2] === 'lin' ? lin : ang).getComponent(j[3]) + j[4]);
    }
    if (p.frame === 'tool') {
      lin.applyQuaternion(sp.quat);
      ang.applyQuaternion(sp.quat);
    }
    const tp = sp.targetPos.clone().addScaledVector(lin, p.jogLinSpeed * dt);
    // keep the target on a short leash so a blocked spoon doesn't wind up a huge error
    const lead = tp.clone().sub(sp.pos);
    if (lead.length() > LEAD_LIN) tp.copy(sp.pos).add(lead.setLength(LEAD_LIN));
    let tq = sp.targetQuat.clone();
    if (ang.lengthSq() > 0) {
      const a = M.MathUtils.degToRad(p.jogAngSpeed) * dt;
      const step = new M.Quaternion().setFromAxisAngle(ang.clone().normalize(), a);
      tq = step.multiply(tq);
      const err = sp.quat.angleTo(tq);
      if (err > LEAD_ANG) tq = sp.quat.clone().slerp(tq, LEAD_ANG / err);
    }
    sp.setTarget(tp, tq);
    return true;
  }

  update(fps) {
    const s = this.sim.getState();
    const rows = this.el.querySelectorAll('.readout tr');
    const fill = (row, pos, eul) => {
      const tds = row.querySelectorAll('td');
      [...pos.map((v) => (v * 1000).toFixed(1)), ...eul.map((v) => v.toFixed(1))].forEach((t, i) => (tds[i + 1].textContent = t));
    };
    fill(rows[1], s.spoon.position, s.spoon.eulerXYZdeg);
    fill(rows[2], s.target.position, s.target.eulerXYZdeg);
    this.$('#c_contact').classList.toggle('on', s.contact);
    this.$('#c_blocked').classList.toggle('on', s.blocked);
    this.$('#c_cut').classList.toggle('on', s.cutForceN > 0.5);
    const g = (v) => `${v.toFixed(1)} g`;
    this.$('#kv').innerHTML = [
      ['on spoon', g(s.scoopedGrams)],
      ['eaten', g(s.eatenGrams)],
      ['spilled', g(s.spilledGrams)],
      ['cut force', `${s.cutForceN.toFixed(1)} N`],
      ['speed scale', `${(s.speedScale * 100).toFixed(0)} %`],
      ['tilt', `${s.spoon.tiltDeg.toFixed(0)}°`],
      ...Object.entries(s.bowlGrams).map(([k, v]) => [k, g(v)]),
      ['particles', s.particles],
      ['fps', fps.toFixed(0)],
    ]
      .map(([k, v]) => `<span>${k}</span><b>${v}</b>`)
      .join('');
  }
}
