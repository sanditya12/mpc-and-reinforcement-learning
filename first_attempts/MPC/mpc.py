from omni.isaac.examples.base_sample import BaseSample
import casadi as ca
import numpy as np
from numpy import pi, sin, cos
import random
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.debug_draw import _debug_draw

from omni.isaac.examples.user_examples.mpc_mul_shooting import CasadiConfig, CostFunction, Simulation 


# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

class MPCController(BaseController):
    def __init__(self, prediction_horizon):
        super().__init__(name="my_cool_controller"),
        self.open_loop_wheel_controller = DifferentialController(name="simple_control",wheel_radius=0.3, wheel_base=0.1125)
        self.casadi_config = CasadiConfig(prediction_horizon)
        self.cost_function = CostFunction()
        self.simulation = Simulation(self.casadi_config, self.cost_function)
        self.simulation.compute_solution_symbolically()  
        self.simulation.prep_nlp_prob()
        self.simulation.prep_solver()
        self.simulation.prep_bounds()
        self.simulation.prep_simulation()
        return

    def forward(
        self,
        start_state,
        target_state,
    ):

        command, predicted_states = self.simulation.step(start_state, target_state)

        print(*command)
        command = np.array([*command[0], *command[1]])
        print(command)
        predicted_states_xyz = predicted_states
        predicted_states_xyz[2,:] = 0.05 # Replacing omega with z values

        target_state_xyz = target_state
        target_state_xyz[2] = .2 
        return command, predicted_states_xyz


        

class ModelPredictiveControl(BaseSample):
    def __init__(
        self
        ) -> None:
        super().__init__()
        self.prediction_horizon = 50
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.open_loop_wheel_controller = DifferentialController(name="simple_control",wheel_radius=0.03, wheel_base=0.1125)

        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("jetbot")
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        self._jetbot_mpc = MPCController(self.prediction_horizon)
        self._world.add_physics_callback("sending_actions", callback_fn=self.on_step)
        self.draw.clear_points()

        return

    def on_step(self, step_size):
        position, orientation = self._jetbot.get_world_pose()
        position_x, position_y, _ = position
        yaw_angle = quat_to_euler_angles(orientation)[-1]

        start_state = [position_x, position_y, yaw_angle]
        target_state = np.array([2.,2.,0])
        command, predicted_states_xyz = self._jetbot_mpc.forward(start_state, target_state)
        
        self.draw.clear_points()
        self.draw.draw_points([target_state], [(0,1,0,1)], [20])
        for predicted_state in predicted_states_xyz.T:
            self.draw.draw_points([predicted_state], [(1,0,0,1)], [5])

        self._jetbot.apply_action(self.open_loop_wheel_controller.forward(command) )


    async def setup_pre_reset(self):
        self.draw.clear_points()
        return

    async def setup_post_reset(self):
        self.draw.clear_points()
        return

    def world_cleanup(self):
        return

        