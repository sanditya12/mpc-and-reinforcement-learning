from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.pipapi
omni.kit.pipapi.install("casadi")
import time
import carb
import numpy as np
import casadi as ca
from casadi import sin, cos, pi
from omni.isaac.core.utils.rotations import quat_to_euler_angles


class MPCComponent():
    mpc_iter = 0

    Q_x = 5
    Q_y = 5
    Q_theta = 5
    R_v = 0.1
    R_omega = 0.1
    v_max = 0.6
    v_min = -v_max
    omega_max = pi / 4
    omega_min = -omega_max
    def __init__(self, N=10, sim_time = 20, step_horizon = 0.1):
        self.N = N
        self.sim_time = sim_time
        self.step_horizon = step_horizon

    def set_weights(self, Q_x, Q_y, Q_theta, R_v, R_omega):
        self.Q_x = Q_x
        self.Q_y = Q_y
        self.Q_theta = Q_theta
        self.R_v = R_v
        self.R_omega = R_omega

    def set_velocity_bounds(self, v_max, v_min, omega_max, omega_min):
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max 
        self.omega_min = omega_min

    def DM2Arr(dm):
        return np.array(dm.full())
    
    def start(self):
        # State Symbolic Variables
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")
        states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = states.numel()
        
        # Control Symbolic Variables
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        controls = ca.vertcat(self.v, self.omega)
        self.n_controls = controls.numel()

        # Matrix containing all states over all time steps + 1 (since it is initial + predictions)
        X = ca.SX.sym("X", self.n_states, self.N + 1)
        
        # Matrix containing all control actions predictions
        U = ca.SX.sym("U", self.n_controls, self.N)
        
        # Parameter vector containing initial and target states
        P = ca.SX.sym("P", self.n_states + self.n_states)
        
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        
        # controls weights matrix
        R = ca.diagcat(self.R_v, self.R_omega)
        
        # Basic System Mapper Function
        rhs = ca.vertcat(self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega)  # right hand side
        f = ca.Function("f", [states, controls], [rhs])

        # Loop for defining objectve function,
        cost_fn = 0
        g = X[:, 0] - P[:self.n_states]  # first constraint element
        
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            cost_fn += (st - P[self.n_states:]).T @ Q @ (st - P[self.n_states:]) + con.T @ R @ con
            st_next = X[:, k + 1]
            st_next_euler = st + (self.step_horizon * f(st, con))
            g = ca.vertcat(g, st_next - st_next_euler)

        # Preparing the NLP
        OPT_variables = ca.vertcat(
            X.reshape(
                (-1, 1)
            ),  # -1 as param means that casadi will automatically find the number of row/columns
            U.reshape((-1, 1)),
        )
        
        nlp_prob = {"f": cost_fn, "x": OPT_variables, "g": g, "p": P}
        
        opts = {
            "ipopt": {
                "max_iter": 2000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }
        
        # Initialize solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
        
        # Initialze Optimization Variables Constraints Vector
        lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
        ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
        
        # Controls Bounds
        lbx[self.n_states * (self.N + 1) :: self.n_controls] = self.v_min  # V lower bound
        lbx[self.n_states * (self.N + 1) + 1 :: self.n_controls] = self.omega_min  # Omega lower bound
        
        ubx[self.n_states * (self.N + 1) :: self.n_controls] = self.v_max  # V upper bound
        ubx[self.n_states * (self.N + 1) + 1 :: self.n_controls] = self.omega_max  # Omega upper bound
        
        self.args = {
            "lbg": ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0
            "ubg": ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0
            "lbx": lbx,
            "ubx": ubx,
        }

        
        
        return

    def step(self, init, target):
        # self.state_init = ca.DM([init[0], init[1], quat_to_euler_angles(init[2])[-1]])
        # self.state_target = ca.DM([target[0], target[1], quat_to_euler_angles(target[2])[-1]])
        self.state_init = ca.DM([init[0], init[1], init[2]])
        self.state_target = ca.DM([target[0], target[1], target[2]])

        if self.mpc_iter == 0 :
            self.u0 = ca.DM.zeros(self.N, self.n_controls)
            self.X0 = ca.repmat(self.state_init, 1, self.N)
        
        if (ca.norm_2(self.state_init - self.state_target) > 1e-1) and (
            self.mpc_iter * self.step_horizon < self.sim_time
        ):
            self.args["p"] = ca.vertcat(self.state_init, self.state_target)
            self.args["x0"] = ca.vertcat(
                ca.reshape(self.X0, self.n_states*(self.N+1)),
                ca.reshape(self.u0, self.n_controls*self.N,1)
            )
            sol = self.solver(
                x0=self.args["x0"],
                lbx=self.args["lbx"],
                ubx=self.args["ubx"],
                lbg=self.args["lbg"],
                ubg=self.args["ubg"],
                p=self.args["p"],
            )
            self.x_sol = ca.reshape(sol["x"][: self.n_states * (self.N + 1)], self.n_states, self.N + 1)
            self.u_sol = ca.reshape(sol["x"][self.n_states * (self.N + 1) :], self.n_controls, self.N)

            self.X0 = ca.horzcat(self.x_sol[:, 1:], self.x_sol[:, -1])
            self.u0 = ca.horzcat(self.u_sol[:, 1:], ca.reshape(self.u_sol[:, -1], -1, 1))
            self.mpc_iter += 1
        
        return self.DM2Arr(self.u0)
        



        self.R_omega = R_omega

    def set_velocity_bounds(self, v_max, v_min, omega_max, omega_min):
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max 
        self.omega_min = omega_min

    def DM2Arr(dm):
        return np.array(dm.full())
    
    def start(self):
        # State Symbolic Variables
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")
        states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = states.numel()
        
        # Control Symbolic Variables
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        controls = ca.vertcat(self.v, self.omega)
        self.n_controls = controls.numel()

        # Matrix containing all states over all time steps + 1 (since it is initial + predictions)
        X = ca.SX.sym("X", self.n_states, self.N + 1)
        
        # Matrix containing all control actions predictions
        U = ca.SX.sym("U", self.n_controls, self.N)
        
        # Parameter vector containing initial and target states
        P = ca.SX.sym("P", self.n_states + self.n_states)
        
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        
        # controls weights matrix
        R = ca.diagcat(self.R_v, self.R_omega)
        
        # Basic System Mapper Function
        rhs = ca.vertcat(self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega)  # right hand side
        f = ca.Function("f", [states, controls], [rhs])

        # Loop for defining objectve function,
        cost_fn = 0
        g = X[:, 0] - P[:self.n_states]  # first constraint element
        
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            cost_fn += (st - P[self.n_states:]).T @ Q @ (st - P[self.n_states:]) + con.T @ R @ con
            st_next = X[:, k + 1]
            st_next_euler = st + (self.step_horizon * f(st, con))
            g = ca.vertcat(g, st_next - st_next_euler)

        # Preparing the NLP
        OPT_variables = ca.vertcat(
            X.reshape(
                (-1, 1)
            ),  # -1 as param means that casadi will automatically find the number of row/columns
            U.reshape((-1, 1)),
        )
        
        nlp_prob = {"f": cost_fn, "x": OPT_variables, "g": g, "p": P}
        
        opts = {
            "ipopt": {
                "max_iter": 2000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }
        
        # Initialize solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
        
        # Initialze Optimization Variables Constraints Vector
        lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
        ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N), 1)
        
        # Controls Bounds
        lbx[self.n_states * (self.N + 1) :: self.n_controls] = self.v_min  # V lower bound
        lbx[self.n_states * (self.N + 1) + 1 :: self.n_controls] = self.omega_min  # Omega lower bound
        
        ubx[self.n_states * (self.N + 1) :: self.n_controls] = self.v_max  # V upper bound
        ubx[self.n_states * (self.N + 1) + 1 :: self.n_controls] = self.omega_max  # Omega upper bound
        
        self.args = {
            "lbg": ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0
            "ubg": ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0
            "lbx": lbx,
            "ubx": ubx,
        }

        
        
        return

    def step(self, x_init, y_init, theta_init, x_target, y_target, theta_target):
        # self.state_init = ca.DM([init[0], init[1], quat_to_euler_angles(init[2])[-1]])
        # self.state_target = ca.DM([target[0], target[1], quat_to_euler_angles(target[2])[-1]])
        self.state_init = ca.DM([x_init, y_init, theta_init])
        self.state_target = ca.DM([x_target, y_target, theta_target])
        if self.mpc_iter == 0 :
            self.u0 = ca.DM.zeros(self.N, self.n_controls)
            self.X0 = ca.repmat(self.state_init, 1, self.N+1)
        
        if (ca.norm_2(self.state_init - self.state_target) > 1e-1) and (
            self.mpc_iter * self.step_horizon < self.sim_time
        ):
            self.args["p"] = ca.vertcat(self.state_init, self.state_target)
            self.args["x0"] = ca.vertcat(
                ca.reshape(self.X0, self.n_states*(self.N+1), 1),
                ca.reshape(self.u0, self.n_controls*self.N,1)
            )
            sol = self.solver(
                x0=self.args["x0"],
                lbx=self.args["lbx"],
                ubx=self.args["ubx"],
                lbg=self.args["lbg"],
                ubg=self.args["ubg"],
                p=self.args["p"],
            )
            self.x_sol = ca.reshape(sol["x"][: self.n_states * (self.N + 1)], self.n_states, self.N + 1)
            self.u_sol = ca.reshape(sol["x"][self.n_states * (self.N + 1) :], self.n_controls, self.N)

            self.X0 = ca.horzcat(self.x_sol[:, 1:], self.x_sol[:, -1])
            self.u0 = ca.horzcat(self.u_sol[:, 1:], ca.reshape(self.u_sol[:, -1], -1, 1))
            self.mpc_iter += 1
        
        return self.DM2Arr(self.u0)
        


class BasicMpc(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        
        jetbot_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        self._jetbot_position, self._jetbot_orientation = self._jetbot.get_world_pose()

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        # print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof)) # prints 2
        # print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))
        self.mpc = MPCComponent()
        self.mpc.start()
        await self._world.play_async()
        self.x = time.time()
        return

    

    def send_robot_actions(self, step_size):
        self._jetbot_position, self._jetbot_orientation = self._jetbot.get_world_pose()
        position_x, position_y, _ = self._jetbot_position
        print(self.mpc.step(position_x, position_y, quat_to_euler_angles(self._jetbot_orientation)[-1], 1, 1, 0))
        return
        # elapsedTime = time.time() - self.x
        # if(elapsedTime<5):
        #     self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities= [2,1]))    
        #     print("Driving ", elapsedTime, "seconds")
        # else: 
        #     self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities= [0,0])) 
        #     #print("Halting after ", elapsedTime, "seconds")