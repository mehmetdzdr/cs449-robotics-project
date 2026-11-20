# Project additions on top of RAI's scenarios/pandasTable.g (two Pandas: l_ = Robot 1, r_ = Robot 2).
# Loaded by conveyor_sort.sim.scene.load_scene(). Root frames use absolute poses (X), because a
# second file cannot reference frames of the first one as parents.
# Table top is at z = 0.6; table spans x in [-1.15, 1.15], y in [-0.62, 0.62], the extension below up to y = 1.1.
# The robots stand on opposite sides of the belt (#154): their base poses are set in sim/scene.py (ROBOT_BASES).

### table extension on Robot 2's side of the belt (the RAI table ends at y = 0.62)

table_far: {
 X: [0, 0.86, 0.55],
 shape: ssBox, size: [2.3, 0.48, 0.1, 0.02], color: [1.],
 contact: 1
}

### top camera: 3.4 m high with a narrow field of view (little perspective distortion), looking straight down, image x = world +x, image up = world +y

cameraTop: {
 X: "t(0 0.15 3.4) d(180 1 0 0)",
 shape: camera, size: [.1],
 focalLength: 1.75, width: 1280, height: 720, zRange: [.5, 100]
}

### conveyor belt: runs along +x between the robots; top surface at z = 0.62

belt: {
 X: [0, 0.2, 0.61],
 shape: ssBox, size: [2.2, 0.3, 0.02, 0.005], color: [.25, .25, .25],
 contact: 1
}

### target regions (thin pale markers on the table, 0.2 m x 0.2 m, no collision)
# region_A, region_B, region_C: Robot 1's side (y < 0), upstream of it.  bin_gray: Robot 2's side (y > 0.4), downstream
# of it. No region is shared, and each robot carries its boxes away from the other robot's part of the belt.

region_A: { X: [-0.44, -0.45, 0.601], shape: box, size: [.2, .2, .002], color: [1., .85, .85] }
region_B: { X: [-0.66, -0.45, 0.601], shape: box, size: [.2, .2, .002], color: [.85, 1., .85] }
region_C: { X: [-0.55, -0.12, 0.601], shape: box, size: [.2, .2, .002], color: [.85, .85, 1.] }
bin_gray: { X: [ 0.45,  0.80, 0.601], shape: box, size: [.2, .2, .002], color: [.8, .8, .8] }

# buffer (#101): for a blocker that the robot removing it cannot take to its own region. With the robots on opposite
# sides (#154) no table spot is reachable by both, and both reach the same part of the belt, so a blocker is always
# removed by the robot that owns its color: the buffer is not used (regions.BUFFER_ROBOTS is empty).
buffer: { X: [-0.22, -0.45, 0.601], shape: box, size: [.2, .2, .002], color: [.95, .95, .8] }
