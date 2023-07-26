from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulate_code import simulate
 
# Matrix Weights Variables
Q_x = 5
Q_y = 5
Q_theta = 0.2
R_v = 0.1
R_omega = 0.1
 
# Simulation Settings
step_horizon = 0.1
N = 20
sim_time = 20
 
wheel_radius = 1  # wheel radius
 
# specs
x_init = 0
y_init = 0
theta_init = 0
x_target = 5.5
y_target = 5.5
theta_target = pi / 2
 
v_max = 0.6
v_min = -v_max
omega_max = pi / 4
omega_min = -omega_max
x_min = -ca.inf
x_max = ca.inf
y_min = -ca.inf
y_max = ca.inf
theta_max = ca.inf
theta_min = -ca.inf
 
 
def DM2Arr(dm):
    return np.array(dm.full())
 
 
# State Symbolic Variables
x = ca.SX.sym("x")
y = ca.SX.sym("y")
theta = ca.SX.sym("theta")
states = ca.vertcat(x, y, theta)
n_states = states.numel()
 
# Control Symbolic Variables
v = ca.SX.sym("v")
omega = ca.SX.sym("omega")
controls = ca.vertcat(v, omega)
n_controls = controls.numel()
 
 
# Matrix containing all states over all time steps + 1 (since it is initial + predictions)
X = ca.SX.sym("X", n_states, N + 1)
 
# Matrix containing all control actions predictions
U = ca.SX.sym("U", n_controls, N)
 
# Parameter vector containing initial and target states
P = ca.SX.sym("P", n_states + n_states)
 
# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)
 
# controls weights matrix
R = ca.diagcat(R_v, R_omega)
 
# Basic System Mapper Function
rhs = ca.vertcat(v @ cos(theta), v @ sin(theta), omega)  # right hand side
f = ca.Function("f", [states, controls], [rhs])
 
# Loop for defining objectve function,
cost_fn = 0
g = X[:, 0] - P[:n_states]  # first constraint element
 
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn += (st - P[n_states:]).T @ Q @ (st - P[n_states:]) + con.T @ R @ con
    st_next = X[:, k + 1]
    st_next_euler = st + (step_horizon * f(st, con))
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
solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
 
 
# Initialze Optimization Variables Constraints Vector
lbx = ca.DM.zeros((n_states * (N + 1) + n_controls * N), 1)
ubx = ca.DM.zeros((n_states * (N + 1) + n_controls * N), 1)
 
# States Bounds
lbx[0 : n_states * (N + 1) : n_states] = x_min  # X lower bound
lbx[1 : n_states * (N + 1) : n_states] = y_min  # Y lower bound
lbx[2 : n_states * (N + 1) : n_states] = theta_min  # theta lower bound
 
ubx[0 : n_states * (N + 1) : n_states] = x_max  # X upper bound
ubx[1 : n_states * (N + 1) : n_states] = y_max  # Y upper bound
ubx[2 : n_states * (N + 1) : n_states] = theta_max  # theta upper bound
 
# Controls Bounds
lbx[n_states * (N + 1) :: n_controls] = v_min  # V lower bound
lbx[n_states * (N + 1) + 1 :: n_controls] = omega_min  # Omega lower bound
 
ubx[n_states * (N + 1) :: n_controls] = v_max  # V upper bound
ubx[n_states * (N + 1) + 1 :: n_controls] = omega_max  # Omega upper bound
 
args = {
    "lbg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints must equal 0
    "ubg": ca.DM.zeros((n_states * (N + 1), 1)),  # constraints must equal 0
    "lbx": lbx,
    "ubx": ubx,
}
 
 
# The Simulation Loop
t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])  # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state
 
t = ca.DM(t0)
 
u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(
    state_init, 1, N + 1
)  # Initial State Matrix containing the trajectories. In this case it is filled with the initial state replicated to fill the X matrix
 
mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
 
 
def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))
    t0 += step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
    return t0, next_state, u0
 
 
main_loop = time()
while (ca.norm_2(state_init - state_target) > 1e-1) and (
    mpc_iter * step_horizon < sim_time
):
    t1 = time()  #
    args["p"] = ca.vertcat(state_init, state_target)
 
    args["x0"] = ca.vertcat(
        ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1)
    )
 
    sol = solver(
        x0=args["x0"],
        lbx=args["lbx"],
        ubx=args["ubx"],
        lbg=args["lbg"],
        ubg=args["ubg"],
        p=args["p"],
    )
 
    u = ca.reshape(sol["x"][n_states * (N + 1) :], n_controls, N)
    X0 = ca.reshape(sol["x"][: n_states * (N + 1)], n_states, N + 1)
 
    cat_states = np.dstack((cat_states, DM2Arr(X0)))
 
    cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
 
    t = np.vstack((t, t0))
 
    t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)
 
    X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))
 
    t2 = time()
 
    times = np.vstack((times, t2 - t1))
 
    mpc_iter += 1
 
simulate(
    cat_states,
    cat_controls,
    times,
    step_horizon,
    N,
    np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),
    save=False,
)
state_init = ca.DM([x_init, y_init, theta_init])  # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state
 
t = ca.DM(t0)
 
u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(
    state_init, 1, N + 1
)  # Initial State Matrix containing the trajectories. In this case it is filled with the initial state replicated to fill the X matrix
 
mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
 
 
def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))
    t0 += step_horizon
    u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
    return t0, next_state, u0
 
 
main_loop = time()
while (ca.norm_2(state_init - state_target) > 1e-1) and (
    mpc_iter * step_horizon < sim_time
):
    t1 = time()  #
    args["p"] = ca.vertcat(state_init, state_target)
 
    args["x0"] = ca.vertcat(
        ca.reshape(X0, n_states * (N + 1), 1), ca.reshape(u0, n_controls * N, 1)
    )
 
    sol = solver(
        x0=args["x0"],
        lbx=args["lbx"],
        ubx=args["ubx"],
        lbg=args["lbg"],
        ubg=args["ubg"],
        p=args["p"],
    )
 
    u = ca.reshape(sol["x"][n_states * (N + 1) :], n_controls, N)
    X0 = ca.reshape(sol["x"][: n_states * (N + 1)], n_states, N + 1)
 
    cat_states = np.dstack((cat_states, DM2Arr(X0)))
 
    cat_controls = np.vstack((cat_controls, DM2Arr(u[:, 0])))
 
    t = np.vstack((t, t0))
 
    t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)
 
    X0 = ca.horzcat(X0[:, 1:], X0[:, -1])
 
    t2 = time()
 
    times = np.vstack((times, t2 - t1))
 
    mpc_iter += 1
 
simulate(
    cat_states,
    cat_controls,
    times,
    step_horizon,
    N,
    np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),
    save=False,
)