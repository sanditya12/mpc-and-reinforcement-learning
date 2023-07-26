import casadi as ca
from casadi import sin, cos, pi
import numpy as np
from time import time

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_euler_angles

class MPCComponent():
    Q_x = 5
    Q_y = 1
    Q_theta = 1
    R_v = 0.1
    R_omega = 0.1
    v_max = 6
    v_min = -v_max
    omega_max = 6
    omega_min = -omega_max
    def __init__(self, N=20, sim_time = 20, step_horizon = 0.1):
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

    def DM2Arr(self,dm):
        return np.array(dm.full())
    
    def start(self,init, target):
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
        self.f = ca.Function("f", [states, controls], [rhs])

        # Loop for defining objectve function,
        cost_fn = 0
        g = X[:, 0] - P[:self.n_states]  # first constraint element
        
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            cost_fn += (st - P[self.n_states:]).T @ Q @ (st - P[self.n_states:]) + con.T @ R @ con
            st_next = X[:, k + 1]
            st_next_euler = st + (self.step_horizon * self.f(st, con))
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

        # States Bounds
        lbx[0 : self.n_states * (self.N + 1) : self.n_states] = -ca.inf # X lower bound
        lbx[1 : self.n_states * (self.N + 1) : self.n_states] = -ca.inf   # Y lower bound
        lbx[2 : self.n_states * (self.N + 1) : self.n_states] = -ca.inf   # theta lower bound

        ubx[0 : self.n_states * (self.N + 1) : self.n_states] = ca.inf   # X upper bound
        ubx[1 : self.n_states * (self.N + 1) : self.n_states] = ca.inf # Y upper bound
        ubx[2 : self.n_states * (self.N + 1) : self.n_states] = ca.inf  # theta upper bound
        
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

        self.init_args = {
            "x" : init['x'],
            "y" : init['y'],
            "theta" : init['theta'],
        }

        self.target_args = {
            "x" : target['x'],
            "y" : target['y'],
            "theta" : target['theta'],
        }

        self.state_init = ca.DM([self.init_args['x'],self.init_args['y'],self.init_args['theta']])
        self.state_target = ca.DM([self.target_args['x'],self.target_args['y'],self.target_args['theta']])
        self.t0 = 0
        self.t = ca.DM(self.t0)
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(
            self.state_init, 1, self.N + 1
        ) 
        
        self.mpc_iter = 0
        self.mpc_completed = False
        self.cat_states = self.DM2Arr(self.X0)
        self.cat_controls = self.DM2Arr(self.u0[:, 0])
        self.times = np.array([[0]])
        return

    def step(self, state_init):
        u = ca.DM.zeros(self.n_controls, self.N)
        if(ca.norm_2(state_init - self.state_target) > 1e-1):
            t1 = time()
            self.args["p"] = ca.vertcat(state_init, self.state_target)
            self.args["x0"] = ca.vertcat(ca.reshape(self.X0, self.n_states*(self.N+1), 1), ca.reshape(self.u0, self.n_controls*self.N, 1))
            sol = self.solver(
                x0=self.args["x0"],
                lbx=self.args["lbx"],
                ubx=self.args["ubx"],
                lbg=self.args["lbg"],
                ubg=self.args["ubg"],
                p=self.args["p"],
            )

            u = ca.reshape(sol["x"][self.n_states*(self.N+1):], self.n_controls, self.N)
            self.X0 = ca.reshape(sol["x"][: self.n_states*(self.N+1)], self.n_states, self.N+1)
            self.cat_states = np.dstack((self.cat_states, self.DM2Arr(self.X0)))
            self.cat_controls = np.vstack((self.cat_controls, self.DM2Arr(u[:, 0])))
            self.t = np.vstack((self.t, self.t0))
            self.t0 += self.step_horizon
            self.u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1)) #Recycling Previous Controls Predicition
            self.X0 = ca.horzcat(self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1)) #Recycling States Matrix
            t2 = time()
            self.times = np.vstack((self.times,t2 - t1))
            self.mpc_iter += 1
         
        else:
            self.mpc_completed = True

        return u
    
    def get_simulation_params(self):
        return {
            "cat_states" : self.cat_states,
            "cat_controls" : self.cat_controls,
            "times": self.times,
            "step_horizon": self.step_horizon,
            "N" : self.N,
            "p_arr": np.array([self.init_args['x'], self.init_args['y'], self.init_args['theta'], self.target_args['x'], self.target_args['y'], self.target_args['theta']]) 
        }
    


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
        print(self._jetbot_orientation)
        x, y, _ = self._jetbot_position
        init = {
            "x" : x,
            "y" : y,
            "theta" : quat_to_euler_angles(self._jetbot_orientation)[-1]
        }
        target = {
            "x" :  1,
            "y" : 1,
            "theta" : quat_to_euler_angles(self._jetbot_orientation)[-1]
        }
        self.state_init = ca.DM([init["x"], init["y"], init["theta"]])  # initial state
        self.mpc = MPCComponent(sim_time=100)
        self.mpc.start(init,target)
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        await self._world.play_async()
        self.x = time.time()
        return

    

    def send_robot_actions(self, step_size):
        if(self.mpc.mpc_completed != True):
            u = self.mpc.step(self.state_init)
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities= [u[0,0],u[1,0]])) 
            self._jetbot_position, self._jetbot_orientation = self._jetbot.get_world_pose()
            x, y, _ = self._jetbot_position  
            self.state_init = ca.DM([x,y, quat_to_euler_angles(self._jetbot_orientation)[-1]])              
            print(x,y)  
        else: 
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None, joint_efforts=None, joint_velocities= [0,0])) 
        
