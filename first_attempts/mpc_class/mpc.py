import casadi as ca
from casadi import sin, cos, pi
import numpy as np
from time import time

class MPC():

    def __init__(self, N=50, step_horizon = 0.2):
        self.N = N
        self.step_horizon = step_horizon

        # init weight variables
        self.Q_x = 5
        self.Q_y = 5
        self.Q_theta = 5
        self.R_v = 0.1
        self.R_omega = 0.1

        # init velocity bounds
        self.v_max = 0.6
        self.v_min = -self.v_max
        self.omega_max = pi/2
        self.omega_min = -self.omega_max

    # util method
    def DM2Arr(self,dm):
        return np.array(dm.full())

    def init_symbolic_vars(self):
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


    # This method also add state constraints based on runge kutta method
    def get_cost_function(self):
        # Basic System Mapper Function
        rhs = ca.vertcat(self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega)  # right hand side
        self.f = ca.Function("f", [self.states, self.controls], [rhs])
        self.cost_fn = 0
        self.g = self.X[:, 0] - self.P[:self.n_states]  # first constraint element
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            self.cost_fn += (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:]) + con.T @ self.R @ con
            st_next = self.X[:, k + 1]
            st_next_euler = st + (self.step_horizon * self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_euler)

    def get_solver (self):
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
    

    def prepare_step (self, init_state: np.ndarray, target_state: np.ndarray):
        self.mpc_completed = False
        self.mpc_iter = 0

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

        # State Constraints from runge kutta must be 0
        lbg = ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0
        ubg = ca.DM.zeros((self.n_states * (self.N + 1), 1)),  # constraints must equal 0

        # initiate state
        self.state_init = ca.DM(init_state)
        self.state_target = ca.DM(target_state)

        # initial control
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  

        # initial trajectory 
        self.X0 = ca.repmat(
            self.state_init, 1, self.N + 1
        ) 

        self.args = {
            "lbg": lbg,
            "ubg": ubg,
            "lbx": lbx,
            "ubx": ubx,
        }
    
    def prepare_with_sim (self):
        self.t0 = 0
        self.t = ca.DM(self.t0)
        self.cat_states = self.DM2Arr(self.X0)
        self.cat_controls = self.DM2Arr(self.u0[:, 0])
        self.times = np.array([[0]])

    def step_with_sim (self,  state_current: np.ndarray, state_target: np.ndarray):
        u = ca.DM.zeros(self.n_controls, self.N)
        self.state_current = ca.DM(state_current)
        self.state_target = ca.DM(state_target)
        next_state = self.state_current
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
            f_value = self.f(self.state_current, u[:, 0])
            next_state = ca.DM.full(self.state_current + (self.step_horizon * f_value))
         
        else:
            self.mpc_completed = True

        return u, self.DM2Arr(next_state)
    
    def get_simulation_params(self):
        return {
            "cat_states" : self.cat_states,
            "cat_controls" : self.cat_controls,
            "times": self.times,
            "step_horizon": self.step_horizon,
            "N" : self.N,
            "p_arr": np.array([self.init_args['x'], self.init_args['y'], self.init_args['theta'], self.target_args['x'], self.target_args['y'], self.target_args['theta']]) 
        }
