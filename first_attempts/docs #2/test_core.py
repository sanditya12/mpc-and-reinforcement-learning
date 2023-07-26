# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
import carb
import time

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class LearnCore(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()

        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube", # The prim path of the cube in the USD stage
                name="fancy_cube", # The unique name used to retrieve the object from the scene later on
                position=np.array([5, 5, 0]), # Using the current stage units which is in meters by default.
                scale=np.array([15.015, 15.015, 15.015]), # most arguments accept mainly numpy arrays.
                color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
            ))
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            # Use carb to log warnings, errors and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")
        #defined the complete path of the specific asset
        asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

        #this code put the robot/asset to the stage (to isaac)
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")

        #wrapping the specified robot/asset into a superclass called Robot class
        jetbot_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._cube = self._world.scene.get_object("fancy_cube")
        self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info)

        self._jetbot = self._world.scene.get_object("fancy_robot")

        #define and get the controller 
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        #specifically an articulaiton controller which usually takes velocity and position then control all the joints or whatever it is so that it can move according to the input. -> abstraction of applying velocity and stuffs
        

        #basically this call back is every second/step
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)

        self.start_time = time.time()

        # Print info about the jetbot after the first reset is called
        print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof)) # prints 2
        print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))
        return
    
    def send_robot_actions(self, step_size):

        #action - backward. must use 2 column tuple for velocities
        # self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=(-5,-5)))

        #action - turn right. first column is left joint and  second is right joint
        self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=(5,3)))

        #action - stop after 5 seconds. once an action is sent, it won't change until other action is sent.
        print("start_time: ", self.start_time, "   , current_time: ", time.time())
        if time.time()-self.start_time<=5.0:
            print("STILL")
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=(5,3)))
        else:
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities=(0,0)))


    #since this is a physics callback, it should have step_size as parameter
    def print_cube_info(self, step_size):
        position, orientation = self._cube.get_world_pose()
        linear_velocity = self._cube.get_linear_velocity()
        # will be shown on terminal
        # print("Cube position is : " + str(position))
        # print("Cube's orientation is : " + str(orientation))
        # print("Cube's linear velocity is : " + str(linear_velocity))


    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
