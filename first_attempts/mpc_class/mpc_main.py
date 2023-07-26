from mpc import MPC
from simulate import simulate
import numpy as np

init_state = [0,0,0]
target_state = [1,1,0]

mpc = MPC()
mpc.init_symbolic_vars()
mpc.get_cost_function()
mpc.get_solver()
mpc.prepare_step(init_state, target_state)
mpc.prepare_with_sim()

while(mpc.mpc_completed != True):
    u, next_state = mpc.step_with_sim(init_state, target_state)
    init_state = next_state

    

sim_params = mpc.get_simulation_params()
simulate(
    sim_params["cat_states"],
    sim_params["cat_controls"],
    sim_params["times"],
    sim_params["step_horizon"],
    sim_params["N"],
    sim_params["p_arr"],
    save = False
)