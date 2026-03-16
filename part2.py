import random
from collections import defaultdict
random.seed(0)

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

def move(s, a):
    if s == GOAL:
        return GOAL

    x, y = s
    nx, ny = x, y

    if a == "U":
        ny += 1
    elif a == "D":
        ny -= 1
    elif a == "L":
        nx -= 1
    elif a == "R":
        nx += 1

    ns = (nx, ny)
    if not in_bounds(ns):
        ns = s  # bump into wall/block -> stay

    return ns

def step_deterministic(s, a):
    if s == GOAL:
        return GOAL, 0, True

    ns = move(s, a)
    done = (ns == GOAL)
    reward = 10 if done else -1
    return ns, reward, done

def left_right_of(a):
    if a in ("U", "D"):
        return "L", "R"
    else:
        return "U", "D"

def stochastic_transition(s, a):
    if s == GOAL:
        return GOAL

    if a == "U":
        transitions = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif a == "D":
        transitions = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif a == "L":
        transitions = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    elif a == "R":
        transitions = [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    r = random.random()
    cum_prob = 0.0
    for action, p in transitions:
        cum_prob += p
        if r <= cum_prob:
            return move(s, action)

    return s

def step_stochastic(s, a):
    if s == GOAL:
        return GOAL, 0, True

    ns = stochastic_transition(s, a)
    done = (ns == GOAL)
    reward = 10 if done else -1
    return ns, reward, done

def get_transition_probs(s, a):
    if s == GOAL:
        return [(1.0, GOAL, 0)]

    if a == "U":
        transitions = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif a == "D":
        transitions = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif a == "L":
        transitions = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    elif a == "R":
        transitions = [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    outcomes = defaultdict(float)
    rewards = {}

    for action, p in transitions:
        ns = move(s, action)
        done = (ns == GOAL)
        reward = 10 if done else -1
        outcomes[ns] += p
        rewards[ns] = reward

    return [(p, ns, rewards[ns]) for ns, p in outcomes.items()]

def all_states():  # to only iterate over valid states
    states = []
    for x in range(W):
        for y in range(H):
            s = (x, y)
            if s not in BLOCKS:
                states.append(s)
    return states

STATES = all_states()

def render_policy(policy):  # printing of arrows for each state ("X" for blocks and "G" for goal)
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    grid = []
    for y in range(H - 1, -1, -1):
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

    for row in grid:
        print(" ".join(row))

def render_values(V):
    grid = []
    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            s = (x, y)
            if s in BLOCKS:
                row.append("X")
            elif s == GOAL:
                row.append("G")
            else:
                row.append(f"{V[s]: 5.2f}")
        grid.append(row)
    for row in grid:
        print(" ".join(row))

def policy_comparison(policy_a, policy_b):  # % of states where the 2 policies choose same action
    states = [s for s in STATES if s != GOAL]
    matches = 0
    for s in states:
        if policy_a.get(s) == policy_b.get(s):
            matches += 1
    return matches / len(states) * 100.0


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
                val = 0.0
                for p, ns, r in get_transition_probs(s, a):
                    val += p * (r + GAMMA * V[ns])
                if val > best:
                    best = val
            V[s] = best
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break

    pi = {}
    for s in STATES:
        if s == GOAL:
            continue
        best_a, best = None, float("-inf")
        for a in ACTIONS:
            val = 0.0
            for p, ns, r in get_transition_probs(s, a):
                val += p * (r + GAMMA * V[ns])
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
            val = 0.0
            for p, ns, r in get_transition_probs(s, a):
                val += p * (r + GAMMA * V[ns])
            V[s] = val
            delta = max(delta, abs(old - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration():
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
            best_a, best = None, float("-inf")
            for a in ACTIONS:
                val = 0.0
                for p, ns, r in get_transition_probs(s, a):
                    val += p * (r + GAMMA * V[ns])
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

    best = max(Q[(s, a)] for a in ACTIONS)
    for a in ACTIONS:
        if Q[(s, a)] == best:
            return a

def greedy_policy_from_Q(Q):  # after training, for each state choose action of max Q
    pi = {}
    for s in STATES:
        if s == GOAL:
            continue
        best = max(Q[(s, a)] for a in ACTIONS)
        for a in ACTIONS:
            if Q[(s, a)] == best:
                pi[s] = a
                break
    return pi


# -----------------------------
# Task 2: Monte Carlo Control (every-visit)
# -----------------------------
def mc_control(num_episodes=20000, eps=0.1, stochastic=True, max_steps=200):
    Q = defaultdict(float)  # estimated value of taking action a in state s
    N = defaultdict(int)  # no.of times visited state action pair

    success = 0  # how many eps reach the goal
    total_steps = 0  # total no.of steps taken across all eps

    for ep in range(num_episodes):
        s = START  # generate episode starting at START
        episode = []  # list of (s,a,r)
        done = False

        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)

            if stochastic:
                ns, r, done = step_stochastic(s, a)
            else:
                ns, r, done = step_deterministic(s, a)

            episode.append((s, a, r))
            s = ns
            if done:
                break

        total_steps += len(episode)
        if done:
            success += 1

        G = 0.0
        for (s, a, r) in reversed(episode):
            G = r + GAMMA * G
            N[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]  # incremental mean

    pi = greedy_policy_from_Q(Q)
    success_rate = success / num_episodes
    avg_steps = total_steps / num_episodes
    return Q, pi, success_rate, avg_steps

# -----------------------------
# Task 3: Q-learning
# -----------------------------
def q_learning(num_episodes=20000, eps=0.1, alpha=0.1, stochastic=True, max_steps=200):
    Q = defaultdict(float)

    success = 0
    total_steps = 0

    for ep in range(num_episodes):
        s = START
        done = False
        steps = 0

        for t in range(max_steps):
            a = epsilon_greedy(Q, s, eps)

            if stochastic:
                ns, r, done = step_stochastic(s, a)
            else:
                ns, r, done = step_deterministic(s, a)

            target = r + (0 if done else GAMMA * max(Q[(ns, ap)] for ap in ACTIONS))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = ns
            steps += 1

            if done:
                break

        total_steps += steps
        if done:
            success += 1

    pi = greedy_policy_from_Q(Q)
    success_rate = success / num_episodes
    avg_steps = total_steps / num_episodes
    return Q, pi, success_rate, avg_steps

# -----------------------------
# Main: run all tasks
# -----------------------------
if __name__ == "__main__":
    print("=== Part 2 Task 1: Value Iteration ===")
    V_vi, pi_vi, vi_iters = value_iteration()
    print(f"Value Iteration converged in {vi_iters} iterations.")
    render_policy(pi_vi)
    print()

    print("Value function (VI): ")
    render_values(V_vi)
    print()

    print("=== Part 2 Task 1: Policy Iteration ===")
    V_pi, pi_pi, pi_rounds = policy_iteration()
    print(f"Policy Iteration converged in {pi_rounds} improvement rounds.")
    render_policy(pi_pi)
    print()

    print("Value function (PI):")
    render_values(V_pi)
    print()

    STOCHASTIC_RL = True  # used for tasks 2 and 3

    print("=== Part 2 Task 2: Monte Carlo Control (eps=0.1) ===")
    Q_mc, pi_mc, mc_sr, mc_avg_steps = mc_control(num_episodes=20000, eps=0.1, stochastic=STOCHASTIC_RL)
    render_policy(pi_mc)
    print(f"MC success rate: {mc_sr*100: .1f}%, avgsteps: {mc_avg_steps: .1f}")

    print("=== Part 2 Task 3: Q-learning (eps=0.1, alpha=0.1) ===")
    Q_ql, pi_ql, ql_sr, ql_avg_steps = q_learning(num_episodes=20000, eps=0.1, alpha=0.1, stochastic=STOCHASTIC_RL)
    render_policy(pi_ql)
    print(f"Q-learning success rate: {ql_sr*100:.1f}%, avg steps: {ql_avg_steps:.1f}")

    print("=== Policy Comparison ===")
    print(f"MC vs VI : {policy_comparison(pi_mc, pi_vi): .1f}%")
    print(f"QL vs VI : {policy_comparison(pi_ql, pi_vi): .1f}%")
    print(f"QL vs MC : {policy_comparison(pi_ql, pi_mc): .1f}%")
    print(f"VI vs PI : {policy_comparison(pi_vi, pi_pi): .1f}%")