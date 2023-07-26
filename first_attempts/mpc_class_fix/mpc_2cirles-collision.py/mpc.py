import casadi as ca
from casadi import sin, cos, pi
import math
import numpy as np
from time import time
from simulate import simulate
from typing import List, Dict

class MPCComponent():
    Q_x = 5
    Q_y = 5
    Q_theta = 0.2
    R_v = 0.1
    R_omega = 0.1
    v_max = 0.6
    v_min = -v_max
    omega_max = pi / 4
    omega_min = -omega_max
    def __init__(self, N=50, sim_time = 20, step_horizon = 0.1, rob_length = 3):
        self.N = N
        self.sim_time = sim_time
        self.step_horizon = step_horizon
        self.rob_diameter = rob_length/2

    def DM2Arr(self,dm):
        return np.array(dm.full())
    
    def init_symbolic_vars (self):
        # State Symbolic Variables
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()
        
        # Control Symbolic Variables
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.numel()

        # Matrix containing all states over all time steps + 1 (since it is initial + predictions)
        self.X = ca.SX.sym("X", self.n_states, self.N + 1)
        
        # Matrix containing all control actions predictions
        self.U = ca.SX.sym("U", self.n_controls, self.N)
        
        # Parameter vector containing initial and target states
        self.P = ca.SX.sym("P", self.n_states + self.n_states)
        
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        
        # controls weights matrix
        self.R = ca.diagcat(self.R_v, self.R_omega)

    def init_cost_fn_and_g_constraints(self):
        # Basic System Mapper Function
        rhs = ca.vertcat(self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega)  # right hand side
        self.f = ca.Function("f", [self.states, self.controls], [rhs])

        # Loop for defining objectve function,
        self.cost_fn = 0
        self.g = self.X[:, 0] - self.P[:self.n_states]  # first constraint element
        
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            self.cost_fn += (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:]) + con.T @ self.R @ con
            st_next = self.X[:, k + 1]
            st_next_euler = st + (self.step_horizon * self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_euler)
    
    def init_solver(self):
        # Preparing the NLP
        OPT_variables = ca.vertcat(
            self.X.reshape(
                (-1, 1)
            ),  # -1 as param means that casadi will automatically find the number of row/columns
            self.U.reshape((-1, 1)),
        )
        
        nlp_prob = {"f": self.cost_fn, "x": OPT_variables, "g": self.g, "p": self.P}
        
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

    def init_constraint_args(self):
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

        # Obstacles Constraints
        if self.obs_len > 0:
            g_obs_len = (self.N+1) * self.obs_len * 2
            lbg = ca.DM.zeros(g_obs_len,1)
            lbg[0: g_obs_len]=-ca.inf
            self.args["lbg"] = ca.vertcat(self.args["lbg"], lbg)

            ubg = ca.DM.zeros(g_obs_len,1)
            self.args["ubg"] = ca.vertcat(self.args["ubg"], ubg)

    def add_obstacle_constraints(self, obs: List[Dict[str, float]]):
        self.obs = obs
        # print(obs[0])
        self.obs_len = len(obs) 
        for j in range(self.obs_len):
            for k in range(self.N + 1):
                c1_x = self.X[0,k] - ((self.rob_diameter/2)*cos(self.X[2,k]))
                c1_y = self.X[1,k] - ((self.rob_diameter/2)*sin(self.X[2,k]))
                c2_x = self.X[0,k] + ((self.rob_diameter/2)*cos(self.X[2,k]))
                c2_y = self.X[1,k] + ((self.rob_diameter/2)*sin(self.X[2,k]))
                constraint1 =  (self.rob_diameter/2 + obs[j]["diameter"]/2)-ca.sqrt(((c1_x-obs[j]["x"])**2)+((c1_y-obs[j]["y"])**2))
                self.g = ca.vertcat(self.g, constraint1)
                constraint2 =  (self.rob_diameter/2 + obs[j]["diameter"]/2)-ca.sqrt(((c2_x-obs[j]["x"])**2)+((c2_y-obs[j]["y"])**2))
                self.g = ca.vertcat(self.g, constraint2)
        

    def add_obstacle_args(self):
        g_obs_len = (self.N+1) * self.obs_len
        lbg = ca.DM.zeros(g_obs_len,1)
        lbg[0: g_obs_len]=-ca.inf
        self.args["lbg"] = ca.vertcat(self.args["lbg"], lbg)

        ubg = ca.DM.zeros(g_obs_len,1)
        self.args["ubg"] = ca.vertcat(self.args["ubg"], ubg)
        

    def step(self, state_current:np.ndarray, state_target: np.ndarray):
        self.state_current = ca.DM(state_current)
        self.state_target = ca.DM(state_target)
        
        u = ca.DM.zeros(self.n_controls, self.N)
        if(ca.norm_2(self.state_current - self.state_target) > 1e-1):

            self.args["p"] = ca.vertcat(self.state_current, self.state_target)
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
            self.u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1)) #Recycling Previous Controls Predicition
            self.X0 = ca.horzcat(self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1)) #Recycling States Matrix
         
        else:
            self.mpc_completed = True

        return u

    def init_sim_params(self):
        self.cat_states = self.DM2Arr(self.X0)
        self.cat_controls = self.DM2Arr(self.u0[:, 0])
        self.times = np.array([[0]])
        self.t0 = 0
        self.t = ca.DM(self.t0)
        self.mpc_iter = 0

    def prepare_step(self,state_init: np.ndarray):
        # self.init_symbolic_vars()
        # self.init_cost_fn_and_g_constraints()
        # self.init_solver()        
        # self.init_constraint_args()
        
        self.mpc_completed = False

        self.state_init = ca.DM(state_init)
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(
            self.state_init, 1, self.N + 1
        ) 
        
        return

    def step_with_sim_params(self, state_current:np.ndarray, state_target: np.ndarray):
        self.state_current = ca.DM(state_current)
        self.state_target = ca.DM(state_target)
        
        u = ca.DM.zeros(self.n_controls, self.N)
        if(ca.norm_2(self.state_current - self.state_target) > 1e-1):
            t1 = time()
            self.args["p"] = ca.vertcat(self.state_current, self.state_target)
            self.args["x0"] = ca.vertcat(ca.reshape(self.X0, self.n_states*(self.N+1), 1), ca.reshape(self.u0, self.n_controls*self.N, 1))
            sol = self.solver(
                x0=self.args["x0"],
                lbx=self.args["lbx"],
                ubx=self.args["ubx"],
                lbg=self.args["lbg"],
                ubg=self.args["ubg"],
                p=self.args["p"],
            )
            print(self.DM2Arr(sol['g']))

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
            "p_arr": np.array([self.state_init[0], self.state_init[1], self.state_init[2], self.state_target[0], self.state_target[1], self.state_target[2]]),
            "obs": self.obs,
            "rob_diam": self.rob_diameter
        }
    
    def simulate_step_shift(self, u, state_init):
        f_value = self.f(state_init, u[:, 0])
        return ca.DM.full(state_init + (self.step_horizon * f_value))

obs = [
    {
        "x": 2,
        "y": 4,
        "diameter": 2
    },
    {
        "x": 6,
        "y": 4,
        "diameter": 3
    },
    {
        "x": 8,
        "y": 8,
        "diameter": 2
    }
]

mpc = MPCComponent()
state_init = np.array([0,0,0])
state_target = np.array([10,10,pi/2])
mpc.init_symbolic_vars()
mpc.init_cost_fn_and_g_constraints()
mpc.add_obstacle_constraints(obs)
mpc.init_solver()        
mpc.init_constraint_args()

mpc.prepare_step(state_init)
mpc.init_sim_params()
while(mpc.mpc_completed != True):
    u = mpc.step_with_sim_params(state_init,state_target)
    
    state_init = mpc.simulate_step_shift(u, state_init)
    # mpc.mpc_completed = True


sim_params = mpc.get_simulation_params()
simulate(
    sim_params["cat_states"],
    sim_params["cat_controls"],
    sim_params["times"],
    sim_params["step_horizon"],
    sim_params["N"],
    sim_params["p_arr"],
    sim_params["obs"],
    sim_params["rob_diam"],
    save = False
)
