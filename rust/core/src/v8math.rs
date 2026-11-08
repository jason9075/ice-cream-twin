//! `Math.sin`, `Math.cos` and `Math.cbrt` exactly as V8 12.x computes them (fdlibm, from
//! v8/src/base/ieee754.cc; Copyright (C) 1993-2004 Sun Microsystems, Inc. — "Permission to
//! use, copy, modify, and distribute this software is freely granted, provided that this
//! notice is preserved."). Bit-identical results keep the Rust and JS simulations in
//! lock-step; arguments beyond 2^19·π/2 (never used here) fall back to `libm`.
// fdlibm's constants and `x - x` (NaN for inf/NaN) are deliberate; keep them verbatim.
#![allow(clippy::approx_constant, clippy::eq_op, clippy::excessive_precision, clippy::unreadable_literal)]

#[inline]
fn high(x: f64) -> i32 {
    (x.to_bits() >> 32) as i32
}
#[inline]
fn from_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | lo as u64)
}

fn kernel_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;
    let ix = high(x) & 0x7FFF_FFFF;
    if ix < 0x3E40_0000 && (x as i32) == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD3_3333 {
        1.0 - (0.5 * z - (z * r - x * y))
    } else {
        let qx = if ix > 0x3FE9_0000 { 0.28125 } else { from_words((ix - 0x0020_0000) as u32, 0) };
        let iz = 0.5 * z - qx;
        let a = 1.0 - qx;
        a - (iz - (z * r - x * y))
    }
}

fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;
    let ix = high(x) & 0x7FFF_FFFF;
    if ix < 0x3E40_0000 && (x as i32) == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

/// fdlibm __ieee754_rem_pio2 for |x| <= 2^19·(π/2). Returns None for larger arguments.
fn rem_pio2(x: f64) -> Option<(i32, f64, f64)> {
    const NPIO2_HW: [i32; 32] = [
        0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A,
        0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
        0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC,
        0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
    ];
    const INVPIO2: f64 = 6.36619772367581382433e-01;
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    const PIO2_2: f64 = 6.07710050630396597660e-11;
    const PIO2_2T: f64 = 2.02226624879595063154e-21;
    const PIO2_3: f64 = 2.02226624871116645580e-21;
    const PIO2_3T: f64 = 8.47842766036889956997e-32;
    let hx = high(x);
    let ix = hx & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        return Some((0, x, 0.0));
    }
    if ix < 0x4002_D97C {
        if hx > 0 {
            let mut z = x - PIO2_1;
            let (y0, y1);
            if ix != 0x3FF9_21FB {
                y0 = z - PIO2_1T;
                y1 = (z - y0) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y0 = z - PIO2_2T;
                y1 = (z - y0) - PIO2_2T;
            }
            return Some((1, y0, y1));
        } else {
            let mut z = x + PIO2_1;
            let (y0, y1);
            if ix != 0x3FF9_21FB {
                y0 = z + PIO2_1T;
                y1 = (z - y0) + PIO2_1T;
            } else {
                z += PIO2_2;
                y0 = z + PIO2_2T;
                y1 = (z - y0) + PIO2_2T;
            }
            return Some((-1, y0, y1));
        }
    }
    if ix <= 0x4139_21FB {
        let mut t = x.abs();
        let n = (t * INVPIO2 + 0.5) as i32;
        let f = n as f64;
        let mut r = t - f * PIO2_1;
        let mut w = f * PIO2_1T;
        let mut y0;
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y0 = r - w;
        } else {
            let j = ix >> 20;
            y0 = r - w;
            let mut i = j - ((high(y0) >> 20) & 0x7FF);
            if i > 16 {
                t = r;
                w = f * PIO2_2;
                r = t - w;
                w = f * PIO2_2T - ((t - r) - w);
                y0 = r - w;
                i = j - ((high(y0) >> 20) & 0x7FF);
                if i > 49 {
                    t = r;
                    w = f * PIO2_3;
                    r = t - w;
                    w = f * PIO2_3T - ((t - r) - w);
                    y0 = r - w;
                }
            }
        }
        let y1 = (r - y0) - w;
        return Some(if hx < 0 { (-n, -y0, -y1) } else { (n, y0, y1) });
    }
    None
}

pub fn sin(x: f64) -> f64 {
    let ix = high(x) & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7FF0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_sin(y0, y1, 1),
            1 => kernel_cos(y0, y1),
            2 => -kernel_sin(y0, y1, 1),
            _ => -kernel_cos(y0, y1),
        },
        None => libm::sin(x),
    }
}

pub fn cos(x: f64) -> f64 {
    let ix = high(x) & 0x7FFF_FFFF;
    if ix <= 0x3FE9_21FB {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7FF0_0000 {
        return x - x;
    }
    match rem_pio2(x) {
        Some((n, y0, y1)) => match n & 3 {
            0 => kernel_cos(y0, y1),
            1 => -kernel_sin(y0, y1, 1),
            2 => -kernel_cos(y0, y1),
            _ => kernel_sin(y0, y1, 1),
        },
        None => libm::cos(x),
    }
}

pub fn cbrt(x: f64) -> f64 {
    const B1: u32 = 715094163;
    const B2: u32 = 696219795;
    const P0: f64 = 1.87595182427177009643;
    const P1: f64 = -1.88497979543377169875;
    const P2: f64 = 1.621429720105354466140;
    const P3: f64 = -0.758397934778766047437;
    const P4: f64 = 0.145996192886612446982;
    let bits = x.to_bits();
    let mut hx = (bits >> 32) as u32;
    let low = bits as u32;
    let sign = hx & 0x8000_0000;
    hx ^= sign;
    if hx >= 0x7FF0_0000 {
        return x + x;
    }
    let mut t;
    if hx < 0x0010_0000 {
        if (hx | low) == 0 {
            return x;
        }
        t = from_words(0x4350_0000, 0) * x;
        let h = (t.to_bits() >> 32) as u32;
        t = from_words(sign | ((h & 0x7FFF_FFFF) / 3 + B2), 0);
    } else {
        t = from_words(sign | (hx / 3 + B1), 0);
    }
    let mut r = (t * t) * (t / x);
    t *= (P0 + r * (P1 + r * P2)) + ((r * r) * r) * (P3 + r * P4);
    t = f64::from_bits((t.to_bits().wrapping_add(0x8000_0000)) & 0xFFFF_FFFF_C000_0000);
    let s = t * t;
    r = x / s;
    let w = t + t;
    r = (r - t) / (w + r);
    t + t * r
}
