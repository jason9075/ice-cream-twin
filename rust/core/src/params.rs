//! Tunable parameters (the panel and the protocol use the same names in camelCase).

#[derive(Clone, Debug)]
pub struct FluidParams {
    /// m
    pub particle_radius: f64,
    /// kernel radius h = kernel_scale * particle diameter
    pub kernel_scale: f64,
    pub substeps: u32,
    pub iterations: u32,
    pub gravity: f64,
    /// XSPH 0..1
    pub viscosity: f64,
    /// 0..1, pulls neighbours together
    pub cohesion: f64,
    /// 1/s
    pub damping: f64,
    /// 0..1 on contacts
    pub friction: f64,
    pub max_particles: usize,
    /// m/s — below this a touching particle counts as resting
    pub stop_speed: f64,
    /// frames at rest before it freezes into the SDF
    pub rest_frames: u32,
    /// stamp sphere radius / particle radius
    pub stamp_scale: f64,
    /// smooth-min blend (m) when freezing
    pub stamp_smooth: f64,
    /// Claybook's DieWhenHitGround
    pub stamp_on_contact: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillShape {
    Scoops,
    Flat,
    Mound,
}

#[derive(Clone, Debug)]
pub struct IceParams {
    /// kPa — cutting resistance
    pub hardness: f64,
    pub accretion: bool,
    /// m grown per carve step
    pub accretion_dist: f64,
    /// cm³ the spoon can carry
    pub capacity: f64,
    /// deg of tilt at which the scoop slides off
    pub release_angle: f64,
    pub displaced_to_particles: bool,
    /// g/cm³ for the mass readouts
    pub density: f64,
    /// 0..1 Laplacian pass on edited voxels each frame
    pub smoothing: f64,
    pub shape: FillShape,
}

#[derive(Clone, Debug)]
pub struct SpoonParams {
    /// m/s
    pub max_lin_speed: f64,
    /// deg/s
    pub max_ang_speed: f64,
    pub jog_lin_speed: f64,
    pub jog_ang_speed: f64,
    /// N — above this the spoon slows down
    pub max_cut_force: f64,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub fluid: FluidParams,
    pub ice: IceParams,
    pub spoon: SpoonParams,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            particle_radius: 0.00125,
            kernel_scale: 1.8,
            substeps: 2,
            iterations: 3,
            gravity: 9.81,
            viscosity: 0.35,
            cohesion: 0.25,
            damping: 0.6,
            friction: 0.7,
            max_particles: 5000,
            stop_speed: 0.035,
            rest_frames: 5,
            stamp_scale: 1.3,
            stamp_smooth: 0.0012,
            stamp_on_contact: false,
        }
    }
}

impl Default for IceParams {
    fn default() -> Self {
        IceParams {
            hardness: 18.0,
            accretion: true,
            accretion_dist: 0.0010,
            capacity: 28.0,
            release_angle: 115.0,
            displaced_to_particles: true,
            density: 0.6,
            smoothing: 0.5,
            shape: FillShape::Scoops,
        }
    }
}

impl Default for SpoonParams {
    fn default() -> Self {
        SpoonParams { max_lin_speed: 0.12, max_ang_speed: 120.0, jog_lin_speed: 0.05, jog_ang_speed: 60.0, max_cut_force: 20.0 }
    }
}

impl Default for Params {
    fn default() -> Self {
        Params { fluid: FluidParams::default(), ice: IceParams::default(), spoon: SpoonParams::default() }
    }
}
