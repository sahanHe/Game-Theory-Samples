import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ====
dt = 0.5
n_steps = 30
d_vals = np.linspace(0, 100, 51)
v_vals = np.linspace(0, 20, 21)
a_vals = np.linspace(-2, 2, 9)
safe_gap = 5.0
max_iters = 10

# Weights
w_a = 0.2   # acceleration effort
w_v = 0.2   # smoothness (velocity change)
w_c = 20.0  # collision penalty weight

def has_collided(d1, d2, n_steps):
    for l in range(0, n_steps):
        if (abs(d1 - 0) +abs(d2-0)) < safe_gap:
            return 1
    return 0

# ==== Cost function ====
def cost_step(d, v, a, v_prev, d_other):
    smoothness = w_a * (a ** 2) + w_v * (v - v_prev) ** 2
    dist = abs(d - d_other)
    # Penalize strong proximity (when dA ≈ dB)

    collision_penalty = w_c * has_collided(d, d_other, n_steps)
    return smoothness + collision_penalty


# ==== DP optimizer ====
def dp_optimize(d_other_traj):
    nd, nv = len(d_vals), len(v_vals)
    dp = np.full((n_steps, nd, nv), np.inf)
    prev = np.zeros((n_steps, nd, nv, 2), dtype=int)

    # Start from (d=0, v=0)
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

    # Backtrack
    id_curr, iv_curr = np.unravel_index(np.argmin(dp[-1]), dp[-1].shape)
    opt_d, opt_v = [], []
    for t in reversed(range(n_steps)):
        opt_d.append(d_vals[id_curr])
        opt_v.append(v_vals[iv_curr])
        id_curr, iv_curr = prev[t, id_curr, iv_curr]
    return np.array(opt_d[::-1]), np.array(opt_v[::-1])


# ==== Initialization ====
# Start positions slightly apart
dA = np.linspace(60, 0, n_steps)
vA = np.full(n_steps, 8.0)
dB = np.linspace(70, 0, n_steps)  # initially ahead by 10 m
vB = np.full(n_steps, 8.0)

# ==== Alternating optimization ====
for it in range(max_iters):
    dA_prev, dB_prev = dA.copy(), dB.copy()
    
    # Fix B, optimize A
    dA, vA = dp_optimize(dB)
    
    # Fix A, optimize B
    dB, vB = dp_optimize(dA)
    
    diff = max(np.max(np.abs(dA - dA_prev)), np.max(np.abs(dB - dB_prev)))
    print(f"Iteration {it+1}: Δ = {diff:.3f}")
    if diff < 1e-3:
        print("Converged.")
        break


# ==== Visualization ====
t = np.arange(n_steps) * dt

plt.figure(figsize=(10,5))
plt.plot(t, dA, 'r-', label='Vehicle A')
plt.plot(t, dB, 'b-', label='Vehicle B')
plt.fill_between(t, dA - safe_gap, dA + safe_gap, color='red', alpha=0.1, label='Safe zone (A)')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Trajectory Optimization: Collision Avoidance Only')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, vA, 'r--', label='Velocity A')
plt.plot(t, vB, 'b--', label='Velocity B')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocities during Iterative Optimization')
plt.legend()
plt.grid(True)
plt.show()
