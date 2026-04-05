import casadi as ca
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
import os

# --- Configuration ---
num_demos = 30_000
MIN_DIST = 2.5  # Minimum distance between start and goal

# ---------------------------------------------------------------------------
# Physical & numerical constants
# ---------------------------------------------------------------------------
DT            = 0.025           # time step (s)
N             = 200            # horizon length (nodes)  => 5.0 s total
MASS          = 1.0           # kg
GRAV          = -9.81         # m s^-2
IX, IY, IZ    = 0.5, 0.1, 0.3 # kg m^2

# ---------------------------------------------------------------------------
# Cost weights
# ---------------------------------------------------------------------------
W_GOAL_POS    = 1e4
W_GOAL_RATE   = 1e3
W_CTRL        = 1e-1
W_VEL         = 1e-2
W_PATH        = 10.0
W_ORIENT      = 1e2

# ---------------------------------------------------------------------------
# Safety / feasibility bounds
# ---------------------------------------------------------------------------
VEL_MAX       = 5.0
ANGVEL_MAX    = 2.0
U1_MIN, U1_MAX = 0.0, -2 * GRAV * MASS
U_TORQUE_MAX  = 0.1
ANGLE_LIMIT   = np.pi / 4.0

# ---------------------------------------------------------------------------
# Quadrotor dynamics (forward Euler)
# ---------------------------------------------------------------------------
def quadrotor_12d_dynamics(x, u, dt=DT):
    psi, theta, phi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

    xdot = ca.vertcat(
        x_dot,
        y_dot,
        z_dot,
        q*ca.sin(phi)/ca.cos(theta) + r*ca.cos(phi)/ca.cos(theta),
        q*ca.cos(phi) - r*ca.sin(phi),
        p + q*ca.sin(phi)*ca.tan(theta) + r*ca.cos(phi)*ca.tan(theta),
        u1/MASS * (ca.sin(phi)*ca.sin(psi) + ca.cos(phi)*ca.cos(psi)*ca.sin(theta)),
        u1/MASS * (ca.cos(psi)*ca.sin(phi) - ca.cos(phi)*ca.sin(psi)*ca.sin(theta)),
        GRAV + u1/MASS * (ca.cos(phi)*ca.cos(theta)),
        ((IY - IZ) / IX) * q*r + u2 / IX,
        ((IZ - IX) / IY) * p*r + u3 / IY,
        ((IX - IY) / IZ) * p*q + u4 / IZ
    )
    return x + dt * xdot

def running_cost(x_i, x_ip1, u_i):
    diff = ca.vertcat(x_ip1[0:3] - x_i[0:3],
                      x_ip1[9:12] - x_i[9:12])
    cost = W_PATH * ca.sqrt(ca.sumsqr(diff))
    cost += W_VEL * ca.sumsqr(x_i[6:9])
    cost += W_CTRL * ca.sumsqr(u_i)
    return cost

def generate_trajectory(start, goal, obs_center=None, obs_radius=0.3):
    nx, nu = 12, 4
    X = ca.SX.sym('X', nx, N+1)
    U = ca.SX.sym('U', nu, N)

    obj = 0
    g   = []
    glb = []
    gub = []

    for k in range(N):
        x_next = quadrotor_12d_dynamics(X[:,k], U[:,k])
        g.append(X[:,k+1] - x_next)
        glb += [0]*nx
        gub += [0]*nx
        obj += running_cost(X[:,k], X[:,k+1], U[:,k])
        obj += W_ORIENT * ca.sumsqr(X[3:5, k])

        if obs_center is not None:
            dist_sq = ca.sumsqr(X[0:3,k] - ca.vcat(obs_center))
            g.append(dist_sq)
            glb.append(obs_radius**2)
            gub.append(1e9)

    g.append(X[:,0] - start)
    glb += [0]*nx
    gub += [0]*nx
    g.append(X[0:3,-1] - goal[0:3])
    glb += [0,0,0]
    gub += [0,0,0]
    g.append(X[3:6,-1] - goal[3:6])
    glb += [0,0,0]
    gub += [0,0,0]
    g.append(X[6:9,-1] - goal[6:9])
    glb += [0,0,0]
    gub += [0,0,0]
    g.append(X[9:12,-1] - goal[9:12])
    glb += [0,0,0]
    gub += [0,0,0]

    nlp = {'x': ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1))),
           'f': obj,
           'g': ca.vertcat(*g)}

    solver = ca.nlpsol('solver','ipopt', nlp,
                   {'print_time':0,
                    'ipopt': {'max_iter': 4000,
                              'tol': 1e-6,
                              "sb": "yes",
                              'print_level': 0}})

    nvar = (N+1)*nx + N*nu
    lbx  = [-1e9]*nvar
    ubx  = [ 1e9]*nvar
    for k in range(N+1):
        base = k*nx
        lbx[base+2] = -5.0
        ubx[base+2] = 5.0
        for j in [3,4,5]:
            lbx[base+j] = -ANGLE_LIMIT
            ubx[base+j] =  ANGLE_LIMIT
        for j in [6,7,8]:
            lbx[base+j] = -VEL_MAX
            ubx[base+j] =  VEL_MAX
        for j in [9,10,11]:
            lbx[base+j] = -ANGVEL_MAX
            ubx[base+j] =  ANGVEL_MAX
    offset_u = (N+1)*nx
    for k in range(N):
        b = offset_u + k*nu
        lbx[b+0] = U1_MIN
        ubx[b+0] = U1_MAX
        for j in [1,2,3]:
            lbx[b+j] = -U_TORQUE_MAX
            ubx[b+j] =  U_TORQUE_MAX

    x0 = []
    for k in range(N+1):
        alpha = k/float(N)
        pos = start[0:3] + alpha*(goal[0:3]-start[0:3])
        x_guess = np.zeros(12)
        x_guess[0:3] = pos
        x0.extend(x_guess)
    for k in range(N):
        hover_thrust = -GRAV * MASS
        x0.extend([hover_thrust, 0.0, 0.0, 0.0])

    sol = solver(lbx=lbx, ubx=ubx, lbg=glb, ubg=gub, x0=x0)
    stats = solver.stats()
    success = stats['success'] and stats['return_status'] == 'Solve_Succeeded'

    if not success:
        return None, None, False

    sol_x = sol['x'].full().ravel()
    X_sol = sol_x[: (N+1)*nx].reshape((nx, N+1), order='F')
    U_sol = sol_x[(N+1)*nx : ].reshape((nu, N), order='F')
    return X_sol, U_sol, True

def worker_func(args):
    start, goal, obs_center, obs_radius = args
    try:
        X, U, success = generate_trajectory(start, goal, obs_center, obs_radius)
        if success:
            return {'start': start, 'goal': goal, 'states': X, 'controls': U}
        else:
            return None
    except Exception as e:
        print(f"Worker PID {os.getpid()} error: {e}")
        return None

if __name__ == "__main__":
    import sys
    from pathlib import Path

    def find_root(start_path, marker=".root"):
        for parent in [start_path] + list(start_path.parents):
            if (parent / marker).exists():
                return parent
        return start_path

    ROOT_DIR = find_root(Path(__file__).resolve().parent)
    sys.path.append(str(ROOT_DIR))
    from utils import resolve_path

    os.makedirs(resolve_path('data'), exist_ok=True)

    # Set obstacle to None for free-space trajectories
    obs_center = None
    obs_radius = 0.0

    low_bounds = np.array([0.1, 0.1, 0.1])
    high_bounds = np.array([4.9, 4.9, 4.9])

    rng = np.random.default_rng(0)
    task_args = []
    print(f"Preparing {num_demos} random task configurations (Free Space)...")

    for _ in range(num_demos):
        # 1. Sample random start anywhere in the 5m cube
        start_pos = rng.uniform(low=low_bounds, high=high_bounds)
        
        # 2. Sample random goal far enough from start
        while True:
            goal_pos = rng.uniform(low=low_bounds, high=high_bounds)
            dist = np.linalg.norm(goal_pos - start_pos)
            if dist > MIN_DIST:
                break
        
        start = np.zeros(12)
        start[0:3] = start_pos
        goal = np.zeros(12)
        goal[0:3] = goal_pos
        
        # obs_center is now None, generate_trajectory handles this automatically
        task_args.append((start, goal, obs_center, obs_radius))

    demos = []
    num_success = 0
    num_fail = 0
    num_cores = 24
    print(f"Starting parallel generation on {num_cores} cores...")

    with mp.Pool(processes=num_cores) as pool:
        results_iterator = pool.imap(worker_func, task_args)
        with tqdm(total=num_demos, desc='Trajectories') as pbar:
            for result in results_iterator:
                if result is not None:
                    demos.append(result)
                    num_success += 1
                else:
                    num_fail += 1
                pbar.set_description(f'Trajectories (Success: {num_success}, Fail: {num_fail})')
                pbar.update(1)

    if num_success > 0:
        save_path = resolve_path('data/random_data.pt')
        torch.save(demos, save_path)
        print(f'\nSaved {num_success} expert trajectories to {save_path}.')
    else:
        print('\nNo trajectories were successfully generated.')
    print(f'Total attempts: {num_demos} (Failed: {num_fail})')
