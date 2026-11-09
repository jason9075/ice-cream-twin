//! Memory map of the GPU simulation and the generated part of its WGSL.
//!
//! Everything lives in a few storage buffers so every kernel shares one bind group:
//! `W` (f32: grid distances, scratch, particle floats, debts, state), `X` (u32: grid colours, flags,
//! lists, particle ints, neighbours, state), `A` (atomic u32: counters, hash-grid cells, fixed-point owe
//! bins), `PP` (particles packed for streaming) and `PK` (grids packed for streaming); plus the per-frame
//! globals `G` and per-dispatch parameters `P` (a dynamic offset into one uniform buffer).

use icecream_core::fluid::MAX_LUMPS;
use icecream_core::sdf::{bowls, BOWL_SPACING, SPOON};
use icecream_core::Sim;
use std::collections::BTreeMap;

pub const CAP: u32 = 16000;
pub const MAXN: u32 = 48;
pub const TABLE: u32 = 1 << 16;
pub const MAX_SPILLED: u32 = 8000;
pub const MAX_DEBTS: u32 = 256;
pub const BIN: u32 = 3;
pub const VOL_UNIT: f64 = 1e-14;
pub const PASS_BYTES: u64 = 256;

/// Named sections laid out back to back (offsets in elements, each a multiple of 4).
#[derive(Clone, Debug, Default)]
pub struct Sections {
    pub at: BTreeMap<&'static str, u32>,
    pub size: u32,
}

impl Sections {
    fn new(list: &[(&'static str, u32)]) -> Self {
        let mut s = Sections::default();
        for &(k, n) in list {
            s.at.insert(k, s.size);
            s.size += n.div_ceil(4) * 4;
        }
        s
    }
    pub fn get(&self, k: &str) -> u32 {
        *self.at.get(k).unwrap_or_else(|| panic!("no section {k}"))
    }
}

#[derive(Clone, Debug)]
pub struct GridInfo {
    pub o: [f64; 3],
    pub c: f64,
    pub n: [u32; 3],
    pub count: u32,
    pub off: u32,
    pub band: f64,
    pub bins: [u32; 3],
    pub bin_off: u32,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub grids: Vec<GridInfo>,
    pub n: u32,
    pub nc: u32,
    pub maxr: u32,
    pub nbins: u32,
    pub w: Sections,
    pub x: Sections,
    pub a: Sections,
}

/// State registers: ST (f32 in W), STU (u32 in X), CNT (atomic counters in A). Same slots as the WGSL.
pub mod st {
    pub const DISPLACED_ACC: u32 = 0;
    pub const CAPTURE_BUDGET: u32 = 1;
    pub const REMOVED: u32 = 2;
    pub const LOOSE_OWED: u32 = 3;
    pub const SPILLED_VOL: u32 = 4;
    pub const EATEN_VOL: u32 = 5;
    pub const CONTENT_VOL: u32 = 6;
    pub const DEBT_VOL: u32 = 7;
    pub const CREDIT: u32 = 8;
    pub const RESIDUAL: u32 = 12;
    pub const NEG_SETTLE: u32 = 16;
    pub const VOLUME: u32 = 20;
    pub const GAINED: u32 = 24;
    pub const CONTENT_VOL_START: u32 = 25;
    pub const DISPLACED_FRAME: u32 = 26;
    pub const SPOON_TOP: u32 = 27;
    pub const GPU_MS: u32 = 28;
    pub const DEBT_SPILLED: u32 = 29;
    pub const ALL: [(&str, u32); 20] = [
        ("displacedAcc", DISPLACED_ACC), ("captureBudget", CAPTURE_BUDGET), ("removed", REMOVED), ("looseOwed", LOOSE_OWED),
        ("spilledVol", SPILLED_VOL), ("eatenVol", EATEN_VOL), ("contentVol", CONTENT_VOL), ("debtVol", DEBT_VOL),
        ("credit", CREDIT), ("residual", RESIDUAL), ("negSettle", NEG_SETTLE), ("volume", VOLUME), ("gained", GAINED),
        ("contentVolStart", CONTENT_VOL_START), ("displacedFrame", DISPLACED_FRAME), ("spoonTop", SPOON_TOP), ("gpuMs", GPU_MS),
        ("debtSpilled", DEBT_SPILLED), ("_pad0", 30), ("_pad1", 31),
    ];
}
pub mod stu {
    pub const N_DEBTS: u32 = 0;
    pub const SETTLE_FRAMES: u32 = 4;
    pub const RESIDUAL_IDX: u32 = 8;
    pub const REDIST_PENDING: u32 = 12;
    pub const RNG: u32 = 13;
    pub const FRAME: u32 = 14;
    pub const FIRST_COLOR: u32 = 15;
    pub const SPAWN_LEFT: u32 = 16;
    pub const RELEASED: u32 = 17;
    pub const RELEASING: u32 = 18;
    /// the last lump id handed out
    pub const LUMP_NEXT: u32 = 19;
    /// 4 distance + 4 colour "changed this frame" marks (boxes live at 32..64)
    pub const DIRTY: u32 = 64;
    pub const ALL: [(&str, u32); 12] = [
        ("nDebts", N_DEBTS), ("settleFrames", SETTLE_FRAMES), ("residualIdx", RESIDUAL_IDX), ("redistPending", REDIST_PENDING),
        ("rng", RNG), ("frame", FRAME), ("firstColor", FIRST_COLOR), ("spawnLeft", SPAWN_LEFT), ("released", RELEASED),
        ("releasing", RELEASING), ("lumpNext", LUMP_NEXT), ("dirty", DIRTY),
    ];
}
pub mod cnt {
    pub const N: u32 = 0;
    pub const SPILLED: u32 = 1;
    pub const NEW_N: u32 = 2;
    pub const FREEZE_N: u32 = 3;
    pub const ATTEMPTS: u32 = 4;
    pub const FIRST_IDX: u32 = 5;
    pub const SPAWN_OK: u32 = 6;
    pub const EATEN_N: u32 = 7;
    pub const BIN_N: u32 = 8;
    pub const LOOSE: u32 = 9;
    pub const ALL: [(&str, u32); 10] = [
        ("n", N), ("spilled", SPILLED), ("newN", NEW_N), ("freezeN", FREEZE_N), ("attempts", ATTEMPTS),
        ("firstIdx", FIRST_IDX), ("spawnOk", SPAWN_OK), ("eatenN", EATEN_N), ("binN", BIN_N), ("loose", LOOSE),
    ];
}

impl Layout {
    pub fn new(sim: &Sim) -> Self {
        let mut grids = Vec::new();
        let mut off = 0;
        let mut bin_off = 0;
        for (k, g) in sim.grids.iter().chain(std::iter::once(&sim.content)).enumerate() {
            let n = [g.nx as u32, g.ny as u32, g.nz as u32];
            let bins = n.map(|v| v.div_ceil(BIN));
            let mut info = GridInfo { o: [g.ox, g.oy, g.oz], c: g.cell, n, count: g.n as u32, off, band: g.band, bins, bin_off: 0 };
            if k < 3 {
                info.bin_off = bin_off;
                bin_off += bins[0] * bins[1] * bins[2];
            }
            off += g.n as u32;
            grids.push(info);
        }
        let n = off;
        let nc = grids[3].count;
        let maxr = grids.iter().map(|g| g.count).max().unwrap();
        let nbins = bin_off;
        let w = Sections::new(&[
            ("D", n), ("TS", n), ("TF", n), ("DVD", maxr), ("DVC", maxr), ("PD", maxr), ("PC", maxr),
            ("CNV", nc), ("RU0", nc), ("RU1", nc), ("FLOOR", nc), ("CB", nc),
            ("POS", 3 * CAP), ("PREV", 3 * CAP), ("VEL", 3 * CAP), ("DP", 3 * CAP), ("CN", 3 * CAP), ("SV", 3 * CAP),
            ("LAM", CAP), ("FPOS", 3 * CAP), ("SLOW", CAP),
            ("QPOS", 3 * CAP), ("QVEL", 3 * CAP), ("QCN", 3 * CAP), ("QSV", 3 * CAP), ("QSLOW", CAP),
            ("SHAPE", 3 * CAP), ("QSHAPE", 3 * CAP), ("LUMPS", 32 * (MAX_LUMPS as u32 + 1)),
            ("PART", 8192), ("SCANB", 2048), ("SPOS", 3 * MAX_SPILLED), ("DEBTF", 4 * MAX_DEBTS), ("ST", 256),
        ]);
        let x = Sections::new(&[
            ("COL", n), ("FLAG", n), ("LIST", n), ("CSRC", nc), ("CFLAG", nc),
            ("AGE", CAP), ("CONTACT", CAP), ("REST", CAP), ("RGBA", CAP), ("SQ", CAP), ("ALIVE", CAP),
            ("CELL", CAP), ("SORTED", CAP), ("NC", CAP), ("FRZ", CAP), ("IDX", CAP),
            ("QAGE", CAP), ("QCONTACT", CAP), ("QREST", CAP), ("QRGBA", CAP), ("LUMP", CAP), ("QLUMP", CAP),
            ("CS", TABLE + 1), ("NB", MAXN * CAP), ("SRGBA", MAX_SPILLED), ("DEBTU", 3 * MAX_DEBTS), ("STU", 256),
            ("SCANU", maxr), ("PRE", maxr), ("BINL", nbins),
        ]);
        let a = Sections::new(&[("CNT", 64), ("CC", TABLE + 1), ("BINS", nbins), ("BOX", 4 * 8), ("SMB", 4 * 8)]);
        Layout { grids, n, nc, maxr, nbins, w, x, a }
    }

    /// Packed grids for streaming: per grid, distance rows then colour rows, 64 u32 per row.
    pub fn pack_offsets(&self) -> (Vec<u32>, u32) {
        let mut offs = Vec::new();
        let mut at = 0;
        for g in &self.grids {
            offs.push(at);
            at += 2 * g.n[1] * g.n[2] * 64;
        }
        (offs, at)
    }

    /// The generated WGSL prefix: layout constants, grids, bowl and spoon numbers.
    pub fn header(&self) -> String {
        fn f(v: f64) -> String {
            let s = format!("{v:?}");
            if s.contains(['.', 'e']) {
                s
            } else {
                format!("{s}.0")
            }
        }
        let mut s = String::new();
        let mut consts = |prefix: &str, list: &mut dyn Iterator<Item = (&str, u32)>| {
            for (k, v) in list {
                s += &format!("const {prefix}{k}: u32 = {v}u;\n");
            }
        };
        consts("W_", &mut self.w.at.iter().map(|(k, v)| (*k, *v)));
        consts("X_", &mut self.x.at.iter().map(|(k, v)| (*k, *v)));
        consts("A_", &mut self.a.at.iter().map(|(k, v)| (*k, *v)));
        consts("S_", &mut st::ALL.iter().copied());
        consts("SU_", &mut stu::ALL.iter().copied());
        consts("C_", &mut cnt::ALL.iter().copied());
        let b = &bowls()[0];
        let h = &SPOON.handle;
        let g = &self.grids;
        s += &format!(
            "const CAP: u32 = {CAP}u;\nconst MAX_LUMPS: u32 = {MAX_LUMPS}u;\nconst MAXN: u32 = {MAXN}u;\nconst TABLE: u32 = {TABLE}u;\nconst MASK: u32 = {}u;\n\
             const MAX_SPILLED: u32 = {MAX_SPILLED}u;\nconst MAX_DEBTS: u32 = {MAX_DEBTS}u;\nconst NBINS: u32 = {}u;\n\
             const BIN: u32 = {BIN}u;\nconst VOL_UNIT: f32 = {};\nconst NC: u32 = {}u;\nconst BIG: f32 = 1e9;\n\n",
            TABLE - 1,
            self.nbins,
            f(VOL_UNIT),
            self.nc
        );
        for (k, v) in [
            ("BOWL_SPACING", BOWL_SPACING), ("BOWL_CY", b.cy), ("BOWL_R", b.r), ("BOWL_T", b.t), ("BOWL_H", b.h), ("BOWL_W", b.w),
            ("BOWL_INNER", b.inner_r), ("BOWL_RIMY", b.rim_y), ("SP_A", SPOON.a), ("SP_B", SPOON.b), ("SP_C", SPOON.c),
            ("SP_T", SPOON.t), ("SP_HR", SPOON.handle_r), ("GEPS", 4e-5),
        ] {
            s += &format!("const {k}: f32 = {};\n", f(v));
        }
        // spoon-local box outside which smoothing needs no keep-out (sim.rs keep_out_box, a bowl band wider)
        let cg = &self.grids[3];
        let size = [cg.n[0] as f64 * cg.c, cg.n[1] as f64 * cg.c, cg.n[2] as f64 * cg.c];
        let (lo, hi) = icecream_core::sim::keep_out_box(cg.o, size, self.grids[0].band);
        s += &format!("const KEEP_LO: vec3f = vec3f({}, {}, {});\nconst KEEP_HI: vec3f = vec3f({}, {}, {});\n", f(lo[0]), f(lo[1]), f(lo[2]), f(hi[0]), f(hi[1]), f(hi[2]));
        let join = |v: Vec<String>| v.join(", ");
        s += &format!(
            "\nvar<private> GO: array<vec4f, 4> = array({});\n",
            join(g.iter().map(|g| format!("vec4f({}, {}, {}, {})", f(g.o[0]), f(g.o[1]), f(g.o[2]), f(g.c))).collect())
        );
        s += &format!(
            "var<private> GN: array<vec4u, 4> = array({});\n",
            join(g.iter().map(|g| format!("vec4u({}u, {}u, {}u, {}u)", g.n[0], g.n[1], g.n[2], g.off)).collect())
        );
        s += &format!("var<private> GBAND: array<f32, 4> = array({});\n", join(g.iter().map(|g| f(g.band)).collect()));
        s += &format!(
            "var<private> GBINS: array<vec4u, 4> = array({});\n",
            join(g.iter().map(|g| format!("vec4u({}u, {}u, {}u, {}u)", g.bins[0], g.bins[1], g.bins[2], g.bin_off)).collect())
        );
        s += &format!("const NHANDLE: u32 = {}u;\n", h.len());
        s += &format!(
            "var<private> HANDLE: array<vec3f, {}> = array({});\n\n",
            h.len(),
            join(h.iter().map(|p| format!("vec3f({}, {}, {})", f(p[0]), f(p[1]), f(p[2]))).collect())
        );
        s
    }
}

/// The whole shader: generated header, shared helpers, then every kernel.
pub fn shader(l: &Layout) -> String {
    [
        l.header(),
        include_str!("wgsl/header.wgsl").to_string(),
        include_str!("wgsl/grid.wgsl").to_string(),
        include_str!("wgsl/fluid.wgsl").to_string(),
        include_str!("wgsl/lump.wgsl").to_string(),
        include_str!("wgsl/world.wgsl").to_string(),
        include_str!("wgsl/freeze.wgsl").to_string(),
        include_str!("wgsl/ledger.wgsl").to_string(),
        include_str!("wgsl/server.wgsl").to_string(),
    ]
    .join("\n")
}

/// Entry points, in the order they appear.
pub fn entry_points(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = code.lines().peekable();
    while let Some(l) = lines.next() {
        if l.trim_start().starts_with("@compute") {
            while let Some(n) = lines.peek() {
                if let Some(rest) = n.trim_start().strip_prefix("fn ") {
                    out.push(rest.split('(').next().unwrap().trim().to_string());
                    break;
                }
                lines.next();
            }
        }
    }
    out
}
