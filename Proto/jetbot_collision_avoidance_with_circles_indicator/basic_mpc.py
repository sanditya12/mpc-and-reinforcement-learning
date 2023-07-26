import casadi as ca
from casadi import sin, cos, pi
import numpy as np
from time import time
from omni.isaac.examples.user_examples.basic_mpc_class import MPCComponent
import math

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.examples.user_examples.mpc_mul_shooting import CasadiConfig, CostFunction, Simulation
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder

class MPCController(BaseController):
    def __init__(self, prediction_horizon):
        super().__init__(name="mpc_controller"),
        self.mpc = MPCComponent(prediction_horizon)
        self.mpc.init_symbolic_vars()
        self.mpc.init_cost_fn_and_g_constraints()
        return



    def prepare(self, obstacles):
        self.mpc.add_obstacle_constraints(obstacles)
        self.mpc.init_solver()        
        self.mpc.init_constraint_args()


    def forward(
        self,
        start_state:np.ndarray,
        target_state:np.ndarray,
    ):
        u,predicted_states = self.mpc.step(start_state, target_state)
        command = np.array(u[:,0].full())
        command = np.array([*command[0],*command[1]])

        predicted_states_xyz = predicted_states
        predicted_states_xyz[2,:] = 0.05 # Replacing omega with z values

        return command, predicted_states_xyz

class BasicMpc(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.open_loop_wheel_controller = DifferentialController(name="simple_control",wheel_radius=0.03, wheel_base=0.1125)
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 400.0
        self._world_settings["rendering_dt"] = 20.0 / 400.0
        self._enter_toggled = 0
        self._base_command = [0.0, 0.0, 0.0, 0]
        self._event_flag = False
        self.draw = _debug_draw.acquire_debug_draw_interface()
        

        return

    def setup_scene(self):
        world = self.get_world()
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
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

        self.obs = [
            {
                "x": 1.3,
                "y": 0.8,
                "diameter": 0.4
            },{
                "x": 0.5,
                "y": 0.5,
                "diameter": 0.2
            }
            ,{
                "x": 1.3,
                "y": 1.9,
                "diameter": 0.5
            },{
                "x": 1.5,
                "y": 0.1,
                "diameter": 0.3
            },
        ]

        for i,ob in enumerate(self.obs):
            world.scene.add(
               DynamicCylinder(
                prim_path="/World/random_cylinder_"+str(i),
                name="ob"+str(i),
                position=np.array([ob['x'], ob["y"], 0]),
                color=np.array([0, 0, 1.0]),
                radius = ob['diameter']/2
                ) 
            )
        # self._a1 = world.scene.add(Unitree(prim_path="/World/A1", name="A1", position=np.array([0, 0, 0.400])))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("jetbot")
        self.mpc_controller = MPCController(30)
        
        self.mpc_controller.prepare(self.obs)
        self.state_target = np.array([2,2,0])
        self._world.add_physics_callback("sending_actions", callback_fn=self.on_step)
        self.draw.clear_points()
        await self._world.play_async()
        return

    

    def on_step(self, step_size):
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        self._jetbot_position, self._jetbot_orientation = self._jetbot.get_world_pose()
        x_curr, y_curr ,_ = self._jetbot_position
        theta_curr = quat_to_euler_angles(self._jetbot_orientation)[-1]
        self.state_curr = np.array([x_curr,y_curr,theta_curr])


        command, predicted_states_xyz = self.mpc_controller.forward(self.state_curr, self.state_target)

        self.draw.clear_lines()
        self.draw.clear_points()
        self.draw.draw_points([self.state_target], [(0,1,0,1)], [20])
        count = 0
        for predicted_state in predicted_states_xyz.T:
            if(count >= 2):
                self.draw_circle(0.15,predicted_state)
                count = 0
            count += 1

        # for predicted_state in predicted_states_xyz.T:
        #     self.draw.draw_points([predicted_state], [(1,0,0,1)], [5])

        self._jetbot.apply_action(self.open_loop_wheel_controller.forward(np.array(command)))
                  
    def draw_circle(self, radius, center):
        num_segments =20
        angle_increment = 360/num_segments
        control_points = []
        for i in range(num_segments):
            angle = i*angle_increment
            x = center[0] + radius * math.cos(math.radians(angle))
            y = center[1] +radius * math.sin(math.radians(angle))
            z = center[2]
            control_point = (x,y,z)
            control_points.append(control_point)
        self.draw.draw_lines_spline(control_points, (1, 0, 0, 1),0, False)
        
