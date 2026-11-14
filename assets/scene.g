# Project additions on top of RAI's scenarios/pandasTable.g (two Pandas: l_ = Robot 1, r_ = Robot 2).
# Loaded by conveyor_sort.sim.scene.load_scene(). Root frames use absolute poses (X), because a
# second file cannot reference frames of the first one as parents.
# Table top is at z = 0.6; table spans x in [-1.15, 1.15], y in [-0.62, 0.62].

### top camera: 3.4 m high with a narrow field of view (little perspective distortion), looking straight down, image x = world +x, image up = world +y

cameraTop: {
 X: "t(0 0.15 3.4) d(180 1 0 0)",
 shape: camera, size: [.1],
 focalLength: 1.75, width: 1280, height: 720, zRange: [.5, 100]
}

### conveyor belt: runs along +x between/in front of the robots; top surface at z = 0.62

belt: {
 X: [0, 0.2, 0.61],
 shape: ssBox, size: [2.2, 0.3, 0.02, 0.005], color: [.25, .25, .25],
 contact: 1
}
