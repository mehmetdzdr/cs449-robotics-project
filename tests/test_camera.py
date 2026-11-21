import numpy as np
import pytest

from conveyor_sort.interfaces import BoxColor
from conveyor_sort.perception.camera import Camera
from conveyor_sort.sim.conveyor import BOX_RGB, ConveyorConfig
from conveyor_sort.sim.world import World


@pytest.fixture
def world():
    world = World(ConveyorConfig(speed=0.2, spacing=0.3))
    world.run(8.0)
    return world


def test_capture_shapes_and_metadata(world):
    obs = Camera(world.C).capture(world.time)
    assert obs.rgb.shape == (720, 1280, 3) and obs.rgb.dtype == np.uint8
    assert obs.depth.shape == obs.segmentation.shape == (720, 1280)
    assert obs.fxycxy[2:] == pytest.approx([640, 360])
    assert obs.cam_position == pytest.approx([0, 0.15, 3.4])
    assert obs.cam_rotation @ [0, 0, 1] == pytest.approx([0, 0, -1])   # viewing axis points down
    assert obs.timestamp == pytest.approx(8.0)


def test_every_box_on_belt_is_visible_with_its_color(world):
    obs = Camera(world.C).capture()
    names = world.conveyor.on_belt()
    assert len(names) == 6
    for name in names:
        mask = obs.frame_mask(world.C, name)
        assert mask.sum() > 400                       # about 30 x 30 px
        rgb = obs.rgb[mask].mean(axis=0) / 255
        expected = np.array(BOX_RGB[world.conveyor.colors[name]])
        if world.conveyor.colors[name] != BoxColor.GRAY:
            assert np.argmax(rgb) == np.argmax(expected)


def test_depth_of_belt_surface(world):
    obs = Camera(world.C).capture()
    belt_depth = obs.depth[obs.frame_mask(world.C, "belt")]
    assert np.median(belt_depth) == pytest.approx(3.4 - world.conveyor.geometry.z_top, abs=1e-3)


class FakeBot:
    """BotOp stand-in: returns `image` as the latest camera image."""
    def __init__(self):
        self.image = 0

    def getImageAndDepth(self, name):
        return np.full((2, 2, 3), self.image, np.uint8), np.zeros((2, 2), np.float32)

    def getCameraFxycxy(self, name):
        return [1.0, 1.0, 1.0, 1.0]


def test_botop_image_stamps_follow_the_loop():
    """A new BotOp image shows the scene around the previous capture: half a step later in a loop of short steps,
    right at it after the loop was busy (the camera renders nothing new meanwhile), unknown after a very long busy
    time. Repeats keep their stamp."""
    from conveyor_sort.perception.camera import IMAGE_DELAY, MAX_BUSY, MAX_RENDER_WAIT
    world = World()
    bot = FakeBot()
    camera = Camera(world.C, bot=bot, segmentation=False)
    assert camera.capture(1.0).timestamp == float("-inf")            # there before the first capture: age unknown
    bot.image = 1
    assert camera.capture(1.05).timestamp == pytest.approx(1.0 + 0.025 - IMAGE_DELAY)
    assert camera.capture(1.10).timestamp == pytest.approx(1.0 + 0.025 - IMAGE_DELAY)   # the same image
    bot.image = 2                                                    # after a 2 s plan
    assert camera.capture(3.10).timestamp == pytest.approx(1.10 + MAX_RENDER_WAIT / 2 - IMAGE_DELAY)
    bot.image = 3
    assert camera.capture(3.10 + MAX_BUSY + 1).timestamp == float("-inf")
