import pytest
import robotic as ry

from conveyor_sort.physics.botop import check_robotic_version, start_simulation, wait_gripper


def test_robotic_version_supports_simulated_grippers():
    check_robotic_version()


@pytest.fixture(scope="module")
def bot_and_config():
    C = ry.Config()
    C.addFile(ry.raiPath("scenarios/pandasTable.g"))
    bot = start_simulation(C)
    yield bot, C
    del bot


@pytest.mark.parametrize("gripper_id", [0, 1])
def test_both_simulated_grippers_close_and_open(bot_and_config, gripper_id):
    bot, C = bot_and_config
    bot.gripperMove(gripper_id, width=0.0)
    assert wait_gripper(bot, C, gripper_id)
    assert bot.getGripperPos(gripper_id) < 0.005
    bot.gripperMove(gripper_id, width=0.08)
    assert wait_gripper(bot, C, gripper_id)
    assert bot.getGripperPos(gripper_id) > 0.035


def test_simulation_time_advances(bot_and_config):
    bot, C = bot_and_config
    t0 = bot.get_t()
    bot.sync(C, 0.5)
    assert bot.get_t() - t0 == pytest.approx(0.5, abs=0.2)
