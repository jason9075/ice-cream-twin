import { Vector3, mat4 } from '../math.js';

/**
 * Orbit camera.
 *   left drag: orbit · right drag / shift+left drag: pan · wheel: zoom
 *   ctrl+left drag: move the spoon target in its horizontal plane · ctrl+wheel: target height
 */
export class OrbitCamera {
  constructor(el, { onTargetDrag, onTargetHeight } = {}) {
    this.el = el;
    this.target = new Vector3(0, 0.07, 0);
    this.distance = 0.62;
    this.azimuth = 0.24; // around Y, 0 = looking from +Z
    this.polar = 0.98; // from +Y
    this.fov = (38 * Math.PI) / 180;
    this.near = 0.005;
    this.far = 10;
    this.onTargetDrag = onTargetDrag;
    this.onTargetHeight = onTargetHeight;
    this._aspect = 1;

    let mode = null;
    let lx = 0;
    let ly = 0;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      lx = e.clientX;
      ly = e.clientY;
      if (e.button === 0 && (e.ctrlKey || e.metaKey) && this.onTargetDrag) {
        mode = 'target';
        this.onTargetDrag('start', this.ray(e.clientX, e.clientY));
      } else if (e.button === 2 || (e.button === 0 && e.shiftKey)) mode = 'pan';
      else if (e.button === 0) mode = 'orbit';
    });
    el.addEventListener('pointermove', (e) => {
      if (!mode) return;
      const dx = e.clientX - lx;
      const dy = e.clientY - ly;
      lx = e.clientX;
      ly = e.clientY;
      if (mode === 'orbit') {
        this.azimuth -= dx * 0.006;
        this.polar = Math.min(Math.PI - 0.05, Math.max(0.05, this.polar - dy * 0.006));
      } else if (mode === 'pan') {
        const s = (this.distance * Math.tan(this.fov / 2) * 2) / el.clientHeight;
        const { right, up } = this.basis();
        this.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
      } else if (mode === 'target') {
        this.onTargetDrag('move', this.ray(e.clientX, e.clientY));
      }
    });
    const end = () => {
      if (mode === 'target') this.onTargetDrag('end');
      mode = null;
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        if ((e.ctrlKey || e.metaKey) && this.onTargetHeight) {
          this.onTargetHeight(-Math.sign(e.deltaY) * 0.004);
          return;
        }
        this.distance = Math.min(3, Math.max(0.08, this.distance * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
  }

  eye() {
    const s = Math.sin(this.polar);
    return new Vector3(
      this.target.x + this.distance * s * Math.sin(this.azimuth),
      this.target.y + this.distance * Math.cos(this.polar),
      this.target.z + this.distance * s * Math.cos(this.azimuth),
    );
  }

  basis() {
    const eye = this.eye();
    const fwd = new Vector3().subVectors(this.target, eye).normalize();
    const right = new Vector3().crossVectors(fwd, new Vector3(0, 1, 0)).normalize();
    const up = new Vector3().crossVectors(right, fwd);
    return { fwd, right, up, eye };
  }

  /** Place the camera at an eye position looking at a target. */
  lookFrom(eye, target) {
    this.target.copy(target);
    const d = new Vector3().subVectors(eye, target);
    this.distance = d.length();
    this.polar = Math.acos(d.y / this.distance);
    this.azimuth = Math.atan2(d.x, d.z);
  }

  view() {
    return mat4.lookAt(this.eye(), this.target, new Vector3(0, 1, 0));
  }

  projection(aspect) {
    this._aspect = aspect;
    return mat4.perspective(this.fov, aspect, this.near, this.far);
  }

  /** World-space ray through a client pixel. */
  ray(cx, cy) {
    const r = this.el.getBoundingClientRect();
    const nx = ((cx - r.left) / r.width) * 2 - 1;
    const ny = 1 - ((cy - r.top) / r.height) * 2;
    const { fwd, right, up, eye } = this.basis();
    const th = Math.tan(this.fov / 2);
    const dir = new Vector3()
      .copy(fwd)
      .addScaledVector(right, nx * th * (r.width / r.height))
      .addScaledVector(up, ny * th)
      .normalize();
    return { origin: eye, dir };
  }
}
