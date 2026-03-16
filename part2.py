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


def in_bounds(state):
    x, y = state
    return 0 <= x < W and 0 <= y < H and state not in BLOCKS


def move(state, action):
    """
    Bottom-left origin:
    U increases y
    D decreases y
    L decreases x
    R increases x
    """
    if state == GOAL:
        return GOAL

    x, y = state
    nx, ny = x, y

    if action == "U":
        ny += 1
    elif action == "D":
        ny -= 1
    elif action == "L":
        nx -= 1
    elif action == "R":
        nx += 1

    next_state = (nx, ny)

    if not in_bounds(next_state):
        return state  # invalid move -> stay in place

    return next_state


def step_deterministic(state, action):
    """
    Deterministic helper for movement/reward logic.
    """
    if state == GOAL:
        return GOAL, 0, True

    next_state = move(state, action)
    done = (next_state == GOAL)
    reward = 10 if done else -1
    return next_state, reward, done


def left_right_of(action):
    """
    Returns the two perpendicular actions.
    """
    if action in ("U", "D"):
        return "L", "R"
    return "U", "D"


def get_transition_probs(state, action):
    """
    Transition model for planning in Task 1.
    Uses known stochastic dynamics:
    intended action with 0.8,
    perpendicular actions with 0.1 each.
    """
    if state == GOAL:
        return [(1.0, GOAL, 0)]

    left_a, right_a = left_right_of(action)

    candidates = [
        (0.8, action),
        (0.1, left_a),
        (0.1, right_a),
    ]

    outcomes = defaultdict(float)
    rewards = {}

    for prob, actual_action in candidates:
        next_state = move(state, actual_action)
        done = (next_state == GOAL)
        reward = 10 if done else -1

        outcomes[next_state] += prob
        rewards[next_state] = reward

    return [(prob, next_state, rewards[next_state]) for next_state, prob in outcomes.items()]


def stochastic_transition(state, action):
    """
    Environment dynamics for Tasks 2 and 3.
    Unknown to the agent, but embedded in the environment.
    """
    if state == GOAL:
        return GOAL

    if action == "U":
        transitions = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "D":
        transitions = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "L":
        transitions = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    else:  # action == "R"
        transitions = [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    r = random.random()
    cumulative = 0.0

    for actual_action, prob in transitions:
        cumulative += prob
        if r <= cumulative:
            return move(state, actual_action)

    return state


def step_stochastic(state, action):
    """
    Sampled interaction step for RL tasks.
    """
    if state == GOAL:
        return GOAL, 0, True

    next_state = stochastic_transition(state, action)
    done = (next_state == GOAL)
    reward = 10 if done else -1
    return next_state, reward, done


def all_states():
    states = []
    for x in range(W):
        for y in range(H):
            state = (x, y)
            if state not in BLOCKS:
                states.append(state)
    return states


STATES = all_states()


def render_policy(policy):
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            state = (x, y)
            if state in BLOCKS:
                row.append("X")
            elif state == GOAL:
                row.append("G")
            else:
                row.append(arrow.get(policy.get(state, " "), " "))
        print(" ".join(row))


def render_values(V):
    for y in range(H - 1, -1, -1):
        row = []
        for x in range(W):
            state = (x, y)
            if state in BLOCKS:
                row.append("   X   ")
            elif state == GOAL:
                row.append("   G   ")
            else:
                row.append(f"{V[state]:6.2f}")
        print(" ".join(row))


def policy_comparison(policy_a, policy_b):
    states = [s for s in STATES if s != GOAL]
    matches = 0

    for state in states:
        if policy_a.get(state) == policy_b.get(state):
            matches += 1

    return matches / len(states) * 100.0


# -----------------------------
# Task 1: Value Iteration
# -----------------------------
def value_iteration(theta=1e-6):
    V = {s: 0.0 for s in STATES}
    iterations = 0

    while True:
        delta = 0.0
        iterations += 1

        for state in STATES:
            if state == GOAL:
                continue

            old_value = V[state]
            best_value = float("-inf")

            for action in ACTIONS:
                expected_value = 0.0
                for prob, next_state, reward in get_transition_probs(state, action):
                    expected_value += prob * (reward + GAMMA * V[next_state])

                best_value = max(best_value, expected_value)

            V[state] = best_value
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    policy = {}
    for state in STATES:
        if state == GOAL:
            continue

        best_action = None
        best_value = float("-inf")

        for action in ACTIONS:
            expected_value = 0.0
            for prob, next_state, reward in get_transition_probs(state, action):
                expected_value += prob * (reward + GAMMA * V[next_state])

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        policy[state] = best_action

    return V, policy, iterations


# -----------------------------
# Task 1: Policy Iteration
# -----------------------------
def policy_evaluation(policy, V, theta=1e-6):
    while True:
        delta = 0.0

        for state in STATES:
            if state == GOAL:
                continue

            old_value = V[state]
            action = policy[state]

            new_value = 0.0
            for prob, next_state, reward in get_transition_probs(state, action):
                new_value += prob * (reward + GAMMA * V[next_state])

            V[state] = new_value
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    return V


def policy_iteration():
    policy = {s: random.choice(ACTIONS) for s in STATES if s != GOAL}
    V = {s: 0.0 for s in STATES}
    rounds = 0

    while True:
        rounds += 1
        V = policy_evaluation(policy, V)

        stable = True

        for state in STATES:
            if state == GOAL:
                continue

            old_action = policy[state]
            best_action = None
            best_value = float("-inf")

            for action in ACTIONS:
                expected_value = 0.0
                for prob, next_state, reward in get_transition_probs(state, action):
                    expected_value += prob * (reward + GAMMA * V[next_state])

                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action

            policy[state] = best_action
            if best_action != old_action:
                stable = False

        if stable:
            break

    return V, policy, rounds


# -----------------------------
# RL helpers
# -----------------------------
def epsilon_greedy(Q, state, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)

    best_value = max(Q[(state, action)] for action in ACTIONS)

    # deterministic tie-break using ACTIONS order
    for action in ACTIONS:
        if Q[(state, action)] == best_value:
            return action


def greedy_policy_from_Q(Q):
    policy = {}

    for state in STATES:
        if state == GOAL:
            continue

        best_value = max(Q[(state, action)] for action in ACTIONS)

        for action in ACTIONS:
            if Q[(state, action)] == best_value:
                policy[state] = action
                break

    return policy


# -----------------------------
# Task 2: Monte Carlo Control (every-visit)
# -----------------------------
def mc_control(num_episodes=20000, eps=0.1, max_steps=200):
    Q = defaultdict(float)
    visit_count = defaultdict(int)

    success = 0
    total_steps = 0

    for _ in range(num_episodes):
        state = START
        episode = []
        done = False

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = step_stochastic(state, action)
            episode.append((state, action, reward))
            state = next_state

            if done:
                break

        total_steps += len(episode)
        if done:
            success += 1

        G = 0.0
        for state, action, reward in reversed(episode):
            G = reward + GAMMA * G
            visit_count[(state, action)] += 1
            Q[(state, action)] += (G - Q[(state, action)]) / visit_count[(state, action)]

    policy = greedy_policy_from_Q(Q)
    success_rate = success / num_episodes
    avg_steps = total_steps / num_episodes

    return Q, policy, success_rate, avg_steps


# -----------------------------
# Task 3: Q-learning
# -----------------------------
def q_learning(num_episodes=20000, eps=0.1, alpha=0.1, max_steps=200):
    Q = defaultdict(float)

    success = 0
    total_steps = 0

    for _ in range(num_episodes):
        state = START
        done = False
        steps = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = step_stochastic(state, action)

            if done:
                target = reward
            else:
                target = reward + GAMMA * max(Q[(next_state, a)] for a in ACTIONS)

            Q[(state, action)] += alpha * (target - Q[(state, action)])

            state = next_state
            steps += 1

            if done:
                break

        total_steps += steps
        if done:
            success += 1

    policy = greedy_policy_from_Q(Q)
    success_rate = success / num_episodes
    avg_steps = total_steps / num_episodes

    return Q, policy, success_rate, avg_steps


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("=== Part 2 Task 1: Value Iteration ===")
    V_vi, pi_vi, vi_iters = value_iteration()
    print(f"Value Iteration converged in {vi_iters} iterations.")
    render_policy(pi_vi)
    print()

    print("Value function (VI):")
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

    print("=== Part 2 Task 2: Monte Carlo Control (eps=0.1) ===")
    Q_mc, pi_mc, mc_success_rate, mc_avg_steps = mc_control(
        num_episodes=20000,
        eps=0.1,
        max_steps=200
    )
    render_policy(pi_mc)
    print(f"MC success rate: {mc_success_rate * 100:.1f}%, avg steps: {mc_avg_steps:.1f}")
    print()

    print("=== Part 2 Task 3: Q-learning (eps=0.1, alpha=0.1) ===")
    Q_ql, pi_ql, ql_success_rate, ql_avg_steps = q_learning(
        num_episodes=20000,
        eps=0.1,
        alpha=0.1,
        max_steps=200
    )
    render_policy(pi_ql)
    print(f"Q-learning success rate: {ql_success_rate * 100:.1f}%, avg steps: {ql_avg_steps:.1f}")
    print()

    print("=== Policy Comparison ===")
    print(f"MC vs VI : {policy_comparison(pi_mc, pi_vi):.1f}%")
    print(f"QL vs VI : {policy_comparison(pi_ql, pi_vi):.1f}%")
    print(f"QL vs MC : {policy_comparison(pi_ql, pi_mc):.1f}%")
    print(f"VI vs PI : {policy_comparison(pi_vi, pi_pi):.1f}%")