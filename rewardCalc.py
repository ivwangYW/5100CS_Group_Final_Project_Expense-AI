def reward_calculation(x, r, a, curr, next_state, if_travel):
    # [now][next],
    # 0'start', 1'deny', 2'approve', 3'require_more_evidence', 4'submit_for_review'
    time_cost = [[0] * 5 for _ in range(5)]
    fraud_risk = [[0] * 5 for _ in range(5)]
    special_case = [[0] * 5 for _ in range(5)]
    amount_variance = [[0] * 5 for _ in range(5)]
    amount_reward = [[0.0] * 5 for _ in range(5)]

    time_cost[0][4] = -100
    time_cost[0][3] = -50
    time_cost[0][2] = 10
    time_cost[3][2] = 10
    time_cost[4][2] = 10

    fraud_risk[0][4] = 30 * r
    if if_travel:
        special_case[0][3] = 5 * 10

    amount_variance[0][4] = 10

    amount_reward[0][3] = 0.08
    amount_reward[0][4] = 0.2

    return time_cost[curr][next_state] + fraud_risk[curr][next_state] + special_case[curr][next_state] + amount_reward[curr][next_state] * x + \
        amount_variance[curr][next_state] * a
