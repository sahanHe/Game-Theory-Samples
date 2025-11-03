# Step 1: Add modules to provide access to specific libraries and functions
import os # Module provides functions to handle file paths, directories, environment variables
import sys # Module provides access to Python-specific system parameters and functions
import random

import numpy as np
import matplotlib.pyplot as plt

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")



# Step 3: Import Traci module
import traci

# Step 4: Define Sumo configuration
ctrl_vehicle_ids = ["t_y", "t_x"]
depart_delays = {"t_x":0,"t_y":0}#{vid: random.uniform(0, 2) for vid in ctrl_vehicle_ids}
print(ctrl_vehicle_ids)
print(depart_delays)


# ==== Parameters ====
dt = 0.5                            # time step        
n_steps = 25                        # number of steps
d_vals = np.linspace(-60, 60, 121)  # possible distances from intersection
v_vals = np.linspace(0, 20, 21)     # velocity grid
a_vals = np.linspace(-2, 2, 9)
safe_dist = 5                     # "collision zone" around center
max_iters = 10


# Weights
w_a = 1000     # acceleration penalty
w_v = 1000     # velocity change penalty
w_c = 5000.0   # collision penalty
w_f = 40000.0   # final distance (goal reaching) penalty


Sumo_config = [
    'sumo-gui',
    '-c', 'GT_crossing.sumocfg',
    '--step-length', '0.5',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
vehicle_speed = 0
total_speed = 0


# ==== Initialization ====
# np.random.seed(42)
# Random starting distances between 30–50 m from center


# Step 7: Define Functions

# ==== Cost function ====
def cost_step(d, v, a, v_prev, d_other):
    smooth = w_a * (a ** 2) + w_v * (v - v_prev) ** 2

    # Both close to center → collision risk
    # Use Gaussian form for smoother decay
    near_center = np.exp(-((d / (2 * safe_dist)) ** 2)) * np.exp(-((d_other / (2 * safe_dist)) ** 2))

    collision_penalty = w_c * near_center

    print(f"Cost step: d={d}, v={v}, a={a}, v_prev={v_prev}, d_other={d_other}, smooth={smooth}, collision_penalty={collision_penalty}")

    return smooth + collision_penalty

# ==== DP optimizer ====
def dp_optimize(d_other_traj, d0, v0):
    nd, nv = len(d_vals), len(v_vals)
    dp = np.full((n_steps, nd, nv), np.inf)
    prev = np.zeros((n_steps, nd, nv, 2), dtype=int)

    # Start from given distance and velocity
    id0 = np.argmin(abs(d_vals - d0))
    iv0 = np.argmin(abs(v_vals - v0))
    dp[0, id0, iv0] = 0

    for t in range(1, n_steps):
        for id_prev, d_prev in enumerate(d_vals):
            for iv_prev, v_prev in enumerate(v_vals):
                if np.isinf(dp[t-1, id_prev, iv_prev]):
                    continue

                for a in a_vals:
                    d_new = d_prev - (v_prev * dt + 0.5*a*dt*dt)   # <-- allow crossing center (no clamp)
                    v_new = v_prev + a * dt

                    # Skip invalid states
                    if d_new < d_vals[0] or d_new > d_vals[-1]:
                        continue
                    if v_new < v_vals[0] or v_new > v_vals[-1]:
                        continue

                    id_new = np.argmin(abs(d_vals - d_new))
                    iv_new = np.argmin(abs(v_vals - v_new))

                    step_cost = cost_step(d_new, v_new, a, v_prev, d_other_traj[t])
                    new_cost = dp[t-1, id_prev, iv_prev] + step_cost

                    if new_cost < dp[t, id_new, iv_new]:
                        dp[t, id_new, iv_new] = new_cost
                        prev[t, id_new, iv_new] = [id_prev, iv_prev]

    # ---- Final cost: encourage reaching the opposite side ----
    d_goal = -d0  # go to the mirrored position (pass through 0)
    dp[-1] += w_f * ((d_vals - d_goal) ** 2)[:, None]

    # ---- Backtrack optimal path ----
    id_curr, iv_curr = np.unravel_index(np.argmin(dp[-1]), dp[-1].shape)
    opt_d, opt_v = [], []
    for t in reversed(range(n_steps)):
        opt_d.append(d_vals[id_curr])
        opt_v.append(v_vals[iv_curr])
        id_curr, iv_curr = prev[t, id_curr, iv_curr]

    return np.array(opt_d[::-1]), np.array(opt_v[::-1])

def within_ctzone(vid, edge, position_on_edge):

    if edge in ["-E4", "-E0"]:
        return position_on_edge >110
    else:
        return False

delay_set = {}

for veh_id in ctrl_vehicle_ids:
    delay_set[veh_id] = True

vehicle_x_speed = random.uniform(18, 20)
vehicle_y_speed = random.uniform(18, 20)
optimized = False
current_step = 0
completed = False
speech_list_x = []
speech_list_y = []

while traci.simulation.getMinExpectedNumber() > 0:
    sim_time = traci.simulation.getTime()
    traci.simulationStep()
    vehicle_status = {"t_x": False, "t_y": False}

    

    # vehicle_ids = traci.vehicle.getIDList()
    
    for vid in list(traci.vehicle.getIDList()):
        
        traci.vehicle.setSpeedMode(vid, 0b00000)
        if delay_set[vid] :
            # print(delay_set[vid])
            if vid in depart_delays and sim_time < depart_delays[vid]:
                # Freeze the vehicle until its time
                traci.vehicle.setSpeed(vid, 0.0)
            else:
                # Allow it to move
                vehicle_speed = vehicle_x_speed if vid == "t_x" else vehicle_y_speed
                traci.vehicle.setSpeed(vid, vehicle_speed)  # default behavior
                traci.vehicle.setSpeedMode(vid, 0b00000)
                delay_set[vid] = False
        else:
            # print(delay_set[vid])
            route = traci.vehicle.getRoute(vid)                     # List of edge IDs in the route
            current_edge = traci.vehicle.getRoadID(vid)             # Current edge ID
            position_on_edge = traci.vehicle.getLanePosition(vid)   # Position along the lane (in meters)
            route_index = route.index(current_edge) if current_edge in route else -1
            ct_plus = within_ctzone(vid, current_edge, position_on_edge)
            print(f"Vehicle {vid}: on edge {current_edge}, is in ctrl zone : {ct_plus}, with position {position_on_edge}")
            vehicle_status[vid] = ct_plus
    if vehicle_status["t_x"] and vehicle_status["t_y"] and not optimized:
        speed_x = traci.vehicle.getSpeed("t_x")
        speed_y = traci.vehicle.getSpeed("t_y")
        print(f"Both vehicles in control zone: t_x speed={speed_x}, t_y speed={speed_y}")
        d_x = 150 - float(traci.vehicle.getDistance("t_x"))
        d_y = 150 - float(traci.vehicle.getDistance("t_y"))
        speed_x = traci.vehicle.getSpeed("t_x")
        speed_y = traci.vehicle.getSpeed("t_y")
        dA0 = d_x
        dB0 = d_y
        vA0 = speed_x
        vB0 = speed_y
        print(f"Initial distances: dA0={dA0}, dB0={dB0}, vA0={vA0}, vB0={vB0}")



        dA = np.linspace(d_x, -d_x, n_steps)
        vA = np.full(n_steps, speed_x)
        dB = -np.linspace(d_y, -d_y, n_steps)
        vB = np.full(n_steps, speed_y)

        for it in range(max_iters):
            dA_prev, dB_prev = dA.copy(), dB.copy()

            # Fix B, optimize A (B is mirrored, so take abs distances)
            dA, vA = dp_optimize(np.abs(dB), dA0, vA0)

            # Fix A, optimize B (symmetric)
            dB_new, vB = dp_optimize(np.abs(dA), dB0, vB0)
            dB = -dB_new  # mirror back to negative side

            diff = max(np.max(np.abs(dA - dA_prev)), np.max(np.abs(dB - dB_prev)))
            print(f"Iteration {it+1}: Δ = {diff:.3f}")
            if diff < 1e-5:
                print("Converged.")
                break
        optimized = True


        # t = np.arange(n_steps) * dt
        # plt.figure(figsize=(10,4))
        # plt.plot(t, vA, 'r--', label='Velocity A')
        # plt.plot(t, vB, 'b--', label='Velocity B')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Velocity (m/s)')
        # plt.title('Vehicle Velocities During Crossing Optimization')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # Set the optimized speeds for the vehicles

    if optimized and (not completed):
        traci.vehicle.setSpeed("t_x", vA[current_step])
        traci.vehicle.setSpeed("t_y", vB[current_step])
        current_step += 1
        if current_step >= n_steps:
            completed = True
        
    if completed:
        traci.vehicle.setSpeedMode("t_x", 0b011111)
        traci.vehicle.setSpeedMode("t_y", 0b011111)
    speech_list_x.append(traci.vehicle.getSpeed("t_x"))
    speech_list_y.append(traci.vehicle.getSpeed("t_y"))




# Step 9: Close connection between SUMO and Traci
traci.close()