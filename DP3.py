import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ====
dt = 0.5                            # time step        
n_steps = 25                        # number of steps
d_vals = np.linspace(-60, 60, 121) #np.linspace(0, 60, 61)     # possible distances from intersection
v_vals = np.linspace(0, 20, 21)     # velocity grid
a_vals = np.linspace(-2, 2, 9)
safe_dist = 5.0                     # "collision zone" around center
max_iters = 10

# Weights
w_a = 1000   # acceleration penalty
w_v = 1000   # velocity change penalty
w_c = 5000.0  # collision penalty
w_f = 3000.0   # final distance (goal reaching) penalty

# def collision_panalty(d):
    

# ==== Cost function ====
def cost_step(d, v, a, v_prev, d_other):
    smooth = w_a * (a ** 2) + w_v * (v - v_prev) ** 2
    # both close to center → collision risk
    near_center = np.exp(-abs(d)/safe_dist) * np.exp(-abs(d_other)/safe_dist)
    collision_penalty = w_c * near_center
    return smooth + collision_penalty


# ==== DP optimizer ====
def dp_optimize(d_other_traj, d0, v0):
    nd, nv = len(d_vals), len(v_vals)
    dp = np.full((n_steps, nd, nv), np.inf)
    prev = np.zeros((n_steps, nd, nv, 2), dtype=int)

    # start from given distance and velocity
    id0 = np.argmin(abs(d_vals - d0))
    iv0 = np.argmin(abs(v_vals - v0))
    dp[0, id0, iv0] = 0

    for t in range(1, n_steps):
        for id_prev, d_prev in enumerate(d_vals):
            for iv_prev, v_prev in enumerate(v_vals):
                if np.isinf(dp[t-1, id_prev, iv_prev]):
                    continue
                for a in a_vals:
                    d_new = max(0, d_prev - v_prev * dt)   # moving toward center
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

    # Final cost: encourage reaching near center
    dp[-1] += w_f * (d_vals ** 2)[:, None]

    # Backtrack
    id_curr, iv_curr = np.unravel_index(np.argmin(dp[-1]), dp[-1].shape)
    opt_d, opt_v = [], []
    for t in reversed(range(n_steps)):
        opt_d.append(d_vals[id_curr])
        opt_v.append(v_vals[iv_curr])
        id_curr, iv_curr = prev[t, id_curr, iv_curr]
    return np.array(opt_d[::-1]), np.array(opt_v[::-1])


# ==== Initialization ====
np.random.seed(42)
# Random starting distances between 30–50 m from center
dA0 = np.random.uniform(30, 50)
dB0 = np.random.uniform(30, 50)
vA0 = np.random.uniform(5, 10)
vB0 = np.random.uniform(5, 10)

# A approaches from positive side, B from negative side
dA = np.linspace(dA0, 0, n_steps)
vA = np.full(n_steps, vA0)
dB = -np.linspace(dB0, 0, n_steps)
vB = np.full(n_steps, vB0)

# ==== Alternating optimization ====
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


# ==== Visualization ====
t = np.arange(n_steps) * dt

plt.figure(figsize=(10,5))
plt.plot(t, dA, 'r-', label='Vehicle A (→ 0)')
plt.plot(t, dB, 'b-', label='Vehicle B (→ 0 from other side)')
plt.axhline(0, color='k', linestyle='--', label='Intersection center')
plt.fill_between(t, -safe_dist, safe_dist, color='gray', alpha=0.2, label='Collision zone')
plt.xlabel('Time (s)')
plt.ylabel('Distance from center (m)')
plt.title('Crossing Trajectory Optimization with Random Start Distances')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, vA, 'r--', label='Velocity A')
plt.plot(t, vB, 'b--', label='Velocity B')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Vehicle Velocities During Crossing Optimization')
plt.legend()
plt.grid(True)
plt.show()

print(dA)
print(dB)
