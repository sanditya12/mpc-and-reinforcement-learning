# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
import numpy as np
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.controllers import BaseController
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.tasks import BaseTask
import carb
import time

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class CustomController(BaseController):
    def __init__(self):
        super().__init__(name="my_cool_controller")
        # An open loop controller that uses a unicycle model. Cuma masukin spec spec dari rodanya
        self._wheel_radius = 0.03
        self._wheel_base = 0.1125
        return
    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=joint_velocities)
    


class FrankaPlaying(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return
    # Here we setup all the assets that we care about in this task.
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="fancy_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.515, 0.515, 0.515]),
                                            color=np.array([0, 0, 1.0])))
        self._franka = scene.add(Franka(prim_path="/World/Fancy_Franka",
                                        name="fancy_franka"))
        return
    
    #getter yang ngereturn observasi posisi frankanya sama posisi si cubenya (semua yg berhubungan dgn task)
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._franka.get_joint_positions()
        observations = {
            self._franka.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations
    
    #ini kyk useeffect yg dependencynya physic step. guna buat track task achieved belum.
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # Visual Materials are applied by default to the cube
            # in this case the cube has a visual material of type
            # PreviewSurface, we can set its color once the target is reached.
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return
    
    # Literally semua trigger bisa di override dan ditaro semacam "hooknya"
    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return


class LearnCore(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):

        world = self.get_world()
        
        world.add_task(FrankaPlaying(name="my_first_task"))

        return

    async def setup_post_load(self):
        self._world = self.get_world()

        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return



    def physics_step(self, step_size):
        current_observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=current_observations["fancy_cube"]["position"],
            placing_position=current_observations["fancy_cube"]["goal_position"],
            current_joint_positions=current_observations["fancy_franka"]["joint_positions"],
        )
        self._franka.apply_action(actions)

        if self._controller.is_done():
            self._world.pause()
        return



    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def world_cleanup(self):
        return
