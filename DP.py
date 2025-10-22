import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ====
dt = 0.5                 # time step (s)
n_steps = 20             # number of time steps
d_vals = np.linspace(0, 100, 51)  # distance discretization
v_vals = np.linspace(0, 20, 21)   # velocity discretization
a_vals = np.linspace(-2, 2, 9)    # acceleration choices
safe_gap = 5.0                    # minimum safe distance (m)
max_iters = 10                    # alternating optimization iterations

# Reference trajectory (same for both, can differ)
d_ref = np.linspace(0, 100, n_steps)
v_ref = np.full(n_steps, 10.0)

# ==== Weights ====
w_d = 1.0    # distance tracking
w_v = 0.5    # velocity tracking
w_a = 0.1    # acceleration smoothness
w_c = 10.0   # collision penalty weight


# ==== Helper: per-step cost function ====
def cost_step(d, v, a, d_ref, v_ref, d_other):
    tracking = w_d * (d - d_ref)**2 + w_v * (v - v_ref)**2 + w_a * (a**2)
    dist = abs(d - d_other)
    collision = w_c * (max(0, safe_gap - dist))**2
    return tracking + collision


# ==== DP Optimization function ====
def dp_optimize(d_ref, v_ref, d_other_traj):
    nd, nv = len(d_vals), len(v_vals)
    dp = np.full((n_steps, nd, nv), np.inf)
    prev = np.zeros((n_steps, nd, nv, 2), dtype=int)

    # initial state = (d=0, v=0)
    id0 = np.argmin(abs(d_vals - 0))
    iv0 = np.argmin(abs(v_vals - 0))
    dp[0, id0, iv0] = 0

    for t in range(1, n_steps):
        for id_prev, d_prev in enumerate(d_vals):
            for iv_prev, v_prev in enumerate(v_vals):
                if np.isinf(dp[t-1, id_prev, iv_prev]):
                    continue
                for a in a_vals:
                    d_new = d_prev + v_prev * dt
                    v_new = v_prev + a * dt

                    # bound
                    if d_new < d_vals[0] or d_new > d_vals[-1]:
                        continue
                    if v_new < v_vals[0] or v_new > v_vals[-1]:
                        continue

                    id_new = np.argmin(abs(d_vals - d_new))
                    iv_new = np.argmin(abs(v_vals - v_new))

                    step_cost = cost_step(d_new, v_new, a, d_ref[t], v_ref[t], d_other_traj[t])
                    new_cost = dp[t-1, id_prev, iv_prev] + step_cost

                    if new_cost < dp[t, id_new, iv_new]:
                        dp[t, id_new, iv_new] = new_cost
                        prev[t, id_new, iv_new] = [id_prev, iv_prev]

    # Backtrack optimal trajectory
    id_curr, iv_curr = np.unravel_index(np.argmin(dp[-1]), dp[-1].shape)
    opt_d, opt_v = [], []
    for t in reversed(range(n_steps)):
        opt_d.append(d_vals[id_curr])
        opt_v.append(v_vals[iv_curr])
        id_curr, iv_curr = prev[t, id_curr, iv_curr]
    return np.array(opt_d[::-1]), np.array(opt_v[::-1])


# ==== Initialize trajectories ====
dA = np.linspace(0, 100, n_steps)
vA = np.full(n_steps, 8.0)
dB = dA - 10.0  # initially behind by 10 m
vB = np.full(n_steps, 8.0)

# ==== Alternating Optimization ====
for it in range(max_iters):
    dA_prev, dB_prev = dA.copy(), dB.copy()
    
    # Fix B, optimize A
    dA, vA = dp_optimize(d_ref, v_ref, dB)
    
    # Fix A, optimize B
    dB, vB = dp_optimize(d_ref, v_ref, dA)
    
    diff = max(np.max(np.abs(dA - dA_prev)), np.max(np.abs(dB - dB_prev)))
    print(f"Iteration {it+1}: Δ = {diff:.3f}")
    if diff < 1e-3:
        print("Converged.")
        break

# ==== Plot results ====
t = np.arange(n_steps) * dt
plt.figure(figsize=(10,5))
plt.plot(t, dA, 'r-', label='Vehicle A')
plt.plot(t, dB, 'b-', label='Vehicle B')
plt.fill_between(t, dA - safe_gap, dA + safe_gap, color='red', alpha=0.1, label='Safe zone (A)')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Optimized Trajectories (DP Alternation)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, vA, 'r--', label='Velocity A')
plt.plot(t, vB, 'b--', label='Velocity B')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Optimized Velocities')
plt.legend()
plt.grid(True)
plt.show()
