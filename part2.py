import random
from collections import defaultdict

# -----------------------------
# Gridworld definition
# -----------------------------
W, H = 5, 5
START = (0, 0)
GOAL = (4, 4)
BLOCKS = {(2, 1), (2, 3)}  # roadblocks
ACTIONS = ["U", "D", "L", "R"]
GAMMA = 0.9

def in_bounds(s):
    x, y = s
    return 0 <= x < W and 0 <= y < H and (x, y) not in BLOCKS

def step_deterministic(s, a):
    """Deterministic transition used for planning (Task 1) and can also be used for RL if required."""
    if s == GOAL:
        return GOAL, 0, True

    x, y = s
    nx, ny = x, y
    if a == "U": ny -= 1
    elif a == "D": ny += 1
    elif a == "L": nx -= 1
    elif a == "R": nx += 1

    ns = (nx, ny)
    if not in_bounds(ns):
        ns = s  # bump into wall/block -> stay

    done = (ns == GOAL)
    reward = 10 if done else -1
    return ns, reward, done

def left_right_of(a):
    # perpendicular actions
    if a in ("U", "D"):
        return "L", "R"
    else:
        return "U", "D"

def step_stochastic(s, a):
    """
    Stochastic transition (0.8 intended, 0.1 left-perp, 0.1 right-perp).
    Use this for RL tasks if your lab requires stochastic dynamics.
    """
    if s == GOAL:
        return GOAL, 0, True

    r = random.random()
    if r < 0.8:
        chosen = a
    else:
        l, rr = left_right_of(a)
        chosen = l if r < 0.9 else rr

    return step_deterministic(s, chosen)

def all_states():
    states = []
    for x in range(W):
        for y in range(H):
            s = (x, y)
            if s not in BLOCKS:
                states.append(s)
    return states

STATES = all_states()

def render_policy(policy):
    """
    policy: dict state -> action
    Prints arrows grid, X for blocks, G for goal.
    """
    arrow = {"U":"↑","D":"↓","L":"←","R":"→"}
    grid = []
    for y in range(H):
        row = []
        for x in range(W):
            s = (x, y)
            if s in BLOCKS:
                row.append("X")
            elif s == GOAL:
                row.append("G")
            else:
                row.append(arrow.get(policy.get(s, " "), " "))
        grid.append(row)
    # Print with y increasing downward (0 at top)
    for row in grid:
        print(" ".join(row))

# -----------------------------
# Task 1: Value Iteration
# -----------------------------
def value_iteration(theta=1e-6):
    V = {s: 0.0 for s in STATES}
    iters = 0

    while True:
        delta = 0.0
        iters += 1
        for s in STATES:
            if s == GOAL:
                continue
            old = V[s]
            best = float("-inf")
            for a in ACTIONS:
                ns, r, done = step_deterministic(s, a)
                val = r + GAMMA * V[ns]
                if val > best:
                    best = val
            V[s] = best
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break

    # derive greedy policy
    pi = {}
    for s in STATES:
        if s == GOAL:
            continue
        best_a, best = None, float("-inf")
        for a in ACTIONS:
            ns, r, done = step_deterministic(s, a)
            val = r + GAMMA * V[ns]
            if val > best:
                best = val
                best_a = a
        pi[s] = best_a
    return V, pi, iters

# -----------------------------
# Task 1: Policy Iteration
# -----------------------------
def policy_evaluation(pi, V, theta=1e-6):
    while True:
        delta = 0.0
        for s in STATES:
            if s == GOAL:
                continue
            old = V[s]
            a = pi[s]
            ns, r, done = step_deterministic(s, a)
            V[s] = r + GAMMA * V[ns]
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration():
    # init random policy (not for goal)
    pi = {s: random.choice(ACTIONS) for s in STATES if s != GOAL}
    V = {s: 0.0 for s in STATES}
    rounds = 0

    while True:
        rounds += 1
        V = policy_evaluation(pi, V)

        stable = True
        for s in STATES:
            if s == GOAL:
                continue
            old_a = pi[s]
            # improve
            best_a, best = None, float("-inf")
            for a in ACTIONS:
                ns, r, done = step_deterministic(s, a)
                val = r + GAMMA * V[ns]
                if val > best:
                    best = val
                    best_a = a
            pi[s] = best_a
            if best_a != old_a:
                stable = False

        if stable:
            break

    return V, pi, rounds

# -----------------------------
# Helpers for RL tasks
# -----------------------------
def epsilon_greedy(Q, s, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    # argmax with tie-break random
    best = max(Q[(s, a)] for a in ACTIONS)
    best_actions = [a for a in ACTIONS if Q[(s, a)] == best]
    return random.choice(best_actions)

def greedy_policy_from_Q(Q):
    pi = {}
    for s in STATES:
        if s == GOAL:
            continue
        best = max(Q[(s, a)] for a in ACTIONS)
        best_actions = [a for a in ACTIONS if Q[(s, a)] == best]
        pi[s] = random.choice(best_actions)
    return pi

# -----------------------------
# Task 2: Monte Carlo Control (every-visit)
# -----------------------------
def mc_control(num_episodes=20000, eps=0.1, stochastic=False, max_steps=200):
    Q = defaultdict(float)
    N = defaultdict(int)

    step_fn = step_stochastic if stochastic else step_deterministic

    for ep in range(num_episodes):
        # generate episode
        s = START
        episode = []  # list of (s,a,r)
        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            ns, r, done = step_fn(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break

        # compute returns backward
        G = 0.0
        for (s, a, r) in reversed(episode):
            G = r + GAMMA * G
            N[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]  # incremental mean

    pi = greedy_policy_from_Q(Q)
    return Q, pi

# -----------------------------
# Task 3: Q-learning
# -----------------------------
def q_learning(num_episodes=20000, eps=0.1, alpha=0.1, stochastic=False, max_steps=200):
    Q = defaultdict(float)
    step_fn = step_stochastic if stochastic else step_deterministic

    for ep in range(num_episodes):
        s = START
        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            ns, r, done = step_fn(s, a)
            target = r + (0 if done else GAMMA * max(Q[(ns, ap)] for ap in ACTIONS))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = ns
            if done:
                break

    pi = greedy_policy_from_Q(Q)
    return Q, pi

# -----------------------------
# Main: run all tasks
# -----------------------------
if __name__ == "__main__":
    print("=== Part 2 Task 1: Value Iteration ===")
    V_vi, pi_vi, vi_iters = value_iteration()
    print(f"Value Iteration converged in {vi_iters} iterations.")
    render_policy(pi_vi)
    print()

    print("=== Part 2 Task 1: Policy Iteration ===")
    V_pi, pi_pi, pi_rounds = policy_iteration()
    print(f"Policy Iteration converged in {pi_rounds} improvement rounds.")
    render_policy(pi_pi)
    print()

    # If your lab requires stochastic dynamics for RL tasks, set stochastic=True below.
    STOCHASTIC_RL = False

    print("=== Part 2 Task 2: Monte Carlo Control (eps=0.1) ===")
    Q_mc, pi_mc = mc_control(num_episodes=20000, eps=0.1, stochastic=STOCHASTIC_RL)
    render_policy(pi_mc)
    print()

    print("=== Part 2 Task 3: Q-learning (eps=0.1, alpha=0.1) ===")
    Q_ql, pi_ql = q_learning(num_episodes=20000, eps=0.1, alpha=0.1, stochastic=STOCHASTIC_RL)
    render_policy(pi_ql)
    print()