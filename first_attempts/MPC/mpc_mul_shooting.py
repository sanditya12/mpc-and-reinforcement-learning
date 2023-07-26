import casadi as ca
import numpy as np
from numpy import sin, cos, pi
from typing import List
class CostFunction():
    def __init__(
        self,
        Q_diag : List[int] = [10,10,0.01],
        R_diag : List[int] = [.1, .1]
        ):
        self.Q = np.zeros([3,3])
        self.Q[0,0], self.Q[1,1], self.Q[2,2] = Q_diag
        
        self.R = np.zeros([2,2])
        self.R[0,0], self.R[1,1] = R_diag
    def __str__(self):
        return f'''\n\n #QR-Matrices for Cost Function# \n\n Q-Matrix:\n\n {self.Q} \n\n R-Matrix:\n\n{self.R}\n\n '''

    def calc_cost(
        self,
        delta_st,
        con
        ):
        return delta_st.T @ self.Q @ delta_st + con.T @ self.R @ con
    
    
class CasadiConfig():
    def __init__(self, prediction_horizon):        
        self.prediction_horizon = prediction_horizon
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')

        self.v = ca.SX.sym('v')
        self.omega = ca.SX.sym('omega')
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.controls = ca.vertcat(self.v, self.omega)

        self.num_states = self.states.numel()
        self.num_controls = self.controls.numel()

        self.rhs = ca.vertcat(self.v * cos(self.theta), self.v * sin(self.theta), self.omega)
        self.f = ca.Function('f', [self.states, self.controls], [self.rhs])
        self.U = ca.SX.sym('U', self.num_controls, prediction_horizon)
        self.P = ca.SX.sym('P', self.num_states + self.num_states)

        self.X = ca.SX.sym('X', self.num_states, prediction_horizon + 1)



class Simulation():
    def __init__(
        self, 
        casadi_config,
        cost_function,
        step_size = 0.2
    ):
        self.cost_function = cost_function
        self.X = casadi_config.X
        self.U = casadi_config.U
        self.P = casadi_config.P
        self.N = casadi_config.prediction_horizon
        self.f = casadi_config.f
        self.num_controls = casadi_config.num_controls
        self.num_states = casadi_config.num_states
        self.T = step_size

        self.v_max = 0.6
        self.v_min = -0.6
        self.omega_max = pi/2
        self.g = self.X[:,0] - self.P[:self.num_states]
        print(self.g)


    def compute_solution_symbolically(self):
        self.obj = 0
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            delta_st = st - self.P[3:6]
            self.obj += self.cost_function.calc_cost(delta_st, con)


            f_value = self.f(st, con)
            st_next = self.X[:,k+1]
            st_next_euler = st + (self.T * f_value)

            self.g = ca.vertcat(
                self.g,
                st_next - st_next_euler
            )        
        print(f'\n\nSymbolic Solution: \n\n {self.g}\n\n')
        print(f'\n\nObjective Function: \n\n {self.obj}\n\n')



    def compute_objective(self):
        self.obj = 0
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            delta_st = st - self.P[3:6]
            self.obj += self.cost_function.calc_cost(delta_st, con)
        
        print(f'\n\nObjective Function: \n\n {self.obj}\n\n')


   
    def prep_nlp_prob(self):
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),
            self.U.reshape((-1, 1))
            )

        self.nlp_prob = {
            'f' : self.obj,
            'x' : self.OPT_variables,
            'g' : self.g,
            'p' : self.P

        }
        print(f'#Nonlinear Programm#\n\n Objective Function:\n\n {self.obj} \n\n Optimization Variables: \n\n {self.OPT_variables}\n\n Constraints: \n\n {self.g} \n\n State Target Matrix P: \n\n{self.P}')

    def prep_solver(self):
        self.opts = {
            'ipopt' : {
                'max_iter' : 100,
                'print_level' : 0,
                'acceptable_tol' : 1e-8,
                'acceptable_obj_change_tol' : 1e-6
            },
            'print_time' : 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)
        print(f'\n\n#Solver#\n\n {self.solver}')

    def prep_bounds(self):
        lbg = ca.DM.zeros((self.num_states * (self.N + 1), 1))
        ubg = ca.DM.zeros((self.num_states * (self.N + 1), 1))

        lbx = ca.DM.zeros((self.num_states * (self.N + 1) + self.num_controls * self.N), 1)
        ubx = ca.DM.zeros((self.num_states * (self.N + 1) + self.num_controls * self.N), 1)

        lbx[0:self.num_states * (self.N + 1): self.num_states] = -ca.inf
        lbx[1:self.num_states * (self.N + 1): self.num_states] = -ca.inf
        lbx[2:self.num_states * (self.N + 1): self.num_states] = -ca.inf

        ubx[0:self.num_states * (self.N + 1): self.num_states] = ca.inf
        ubx[1:self.num_states * (self.N + 1): self.num_states] = ca.inf
        ubx[2:self.num_states * (self.N + 1): self.num_states] = ca.inf


        lbx[self.num_states * (self.N + 1): : self.num_controls] = self.v_min
        lbx[self.num_states * (self.N + 1) + 1: : self.num_controls] = -self.omega_max
        
        ubx[self.num_states * (self.N + 1): : self.num_controls] = self.v_max
        ubx[self.num_states * (self.N + 1) + 1: : self.num_controls] = self.omega_max
        
        print(f'\n\n#Bounds#\n\n')
        print(f' Lower Bound Equality Constraints:\n\n {lbg}\n\n')
        print(f' Upper Bound Equality Constraints:\n\n {ubg}\n\n')
        print(f' Lower Bound State Constraints:\n\n {lbx}\n\n')
        print(f' Upper Bound State Constraints:\n\n {ubx}\n\n')
        self.args = {
            'lbg' : lbg,
            'ubg' : ubg,
            'lbx' : lbx,
            'ubx' : ubx
        }

    def prep_simulation(self):
        self.u0 = ca.DM.zeros((self.num_controls, self.N))
        self.init_state = ca.DM([0.0, 0.0, 0.0])

        self.X0 = ca.repmat(self.init_state, 1, self.N + 1)


    def step(self,
    init_state : np.ndarray,
    target_state: np.ndarray
    ) -> np.ndarray:
        
        self.init_state = ca.DM(init_state)
        self.target_state = ca.DM(target_state)

        self.args['p'] = ca.vertcat(
            self.init_state,
            self.target_state
        )

        self.args['x0'] = ca.vertcat(
            ca.reshape(self.X0, self.num_states * (self.N + 1), 1),
            ca.reshape(self.u0, self.num_controls * self.N, 1)
        )


        sol = self.solver(
            x0 = self.args['x0'],
            lbx= self.args['lbx'],
            ubx= self.args['ubx'],
            lbg= self.args['lbg'],
            ubg= self.args['ubg'],
            p= self.args['p'] )
        
    
        self.X0 = ca.reshape(sol['x'][ : self.num_states * (self.N + 1)], self.num_states, self.N + 1)
        self.u = ca.reshape(sol['x'][self.num_states * (self.N + 1) : ], self.num_controls, self.N)
        command = np.array(self.u[:, 0].full())
       
        predicted_states = np.array(self.X0.full()[:,1:])
        ss_error = ca.norm_2(self.init_state - self.target_state)
        print(f'ss_error: {ss_error}')
        return command, predicted_states




    
