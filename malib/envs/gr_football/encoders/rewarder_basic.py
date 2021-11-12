import numpy as np
import torch


def calc_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
    if obs["steps_left"] == 0:
        # print("STEPS LEFT == 1!")
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    goal_switch = 0.0
    if prev_obs["ball_owned_team"] == 1 and obs["ball_owned_team"] == 0:
        goal_switch = 1.0
    elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 0:
        goal_switch = -1.0

    reward = 5.0 * win_reward + 5.0 * rew + 0.003 * ball_position_r + yellow_r
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
    else:
        min_dist2ball = 0.0

    return np.array([rew, min_dist2ball, yellow_r, ball_position_r, goal_switch])


def calc_skilled_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    if prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 1:
        ballowned_r = 1.0
    elif prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 0:
        ballowned_r = 0.0
    else:
        ballowned_r = -1.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = (
        2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r + ballowned_r
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    left_team_position = obs["left_team"]
    right_team_position = obs["right_team"]

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)

    reward = (
        2.0 * win_reward
        + 20.0 * rew
        + 0.06 * ball_position_r
        + yellow_r
        + left_position_move
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_skilled_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    if prev_obs["ball_owned_team"] == 1 or prev_obs["ball_owned_team"] == 0:
        if (
            obs["ball_owned_team"] == 1
            and prev_obs["ball_owned_player"] != obs["ball_owned_player"]
        ):
            highpass_r = 2.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = (
        2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r + highpass_r
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_offside_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 5.0 * rew + 0.06 * ball_position_r + 20 * yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward
