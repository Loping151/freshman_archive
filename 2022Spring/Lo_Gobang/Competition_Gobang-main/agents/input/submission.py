def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    __x, __y = map(int, input().split())
    agent_action = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    agent_action[0][__x] = 1
    agent_action[1][__y] = 1
    return agent_action
