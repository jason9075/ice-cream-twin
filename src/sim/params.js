export function defaultParams() {
  return {
    fluid: {
      particleRadius: 0.00125, // m
      kernelScale: 1.8, // kernel radius h = kernelScale * particle diameter
      substeps: 2,
      iterations: 3,
      gravity: 9.81,
      viscosity: 0.35, // XSPH 0..1
      cohesion: 0.25, // 0..1, pulls neighbours together (sticky ice cream)
      damping: 0.6, // 1/s
      friction: 0.7, // 0..1 on contacts
      maxParticles: 5000,
      stopSpeed: 0.035, // m/s — below this a touching particle counts as resting
      restFrames: 5, // frames at rest before it freezes into the SDF
      stampScale: 1.3, // stamp sphere radius / particle radius
      stampSmooth: 0.0012, // smooth-min blend (m) when freezing
      stampOnContact: false, // Claybook's DieWhenHitGround
    },
    ice: {
      hardness: 18, // kPa — cutting resistance, sets the estimated cutting force
      accretion: true, // material ahead of the scoop sticks to it (curling)
      accretionDist: 0.0010, // m grown per carve step
      capacity: 28, // cm³ the spoon can carry before extra becomes crumbs
      releaseAngle: 115, // deg of tilt at which the scoop slides off
      displacedToParticles: true, // metal-displaced ice becomes crumbs (else it is compacted)
      density: 0.6, // g/cm³ for the mass readouts
      smoothing: 0.5, // 0..1 Laplacian pass on edited voxels each frame (surface tension)
      shape: 'scoops',
    },
    spoon: {
      maxLinSpeed: 0.12, // m/s
      maxAngSpeed: 120, // deg/s
      jogLinSpeed: 0.05, // m/s
      jogAngSpeed: 60, // deg/s
      maxCutForce: 20, // N — above this the spoon slows down
      frame: 'world', // jog frame: world | tool
    },
  };
}
