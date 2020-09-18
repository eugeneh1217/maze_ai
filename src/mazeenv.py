import gym
from gym import spaces
import numpy as np
import csv

"""
Squares:
    0 --> blank square
    1 --> barrier
    2 --> point square
    3 --> end
    4 --> player

Actions:
    0 --> don't move
    1 --> move up
    2 --> move right
    3 --> move down
    4 --> move left
"""

MOVEMENT_ARRAY = [
    [0, 0],
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1]
]

COMPLETION_REWARD = 10
STEP_REWARD = -1
POINT_REWARD = 5

DEFAULT_MAZE_ARRAY = []
with open("defaultarray8.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        DEFAULT_MAZE_ARRAY.append(row)

BOARD_RESOLUTION = (len(DEFAULT_MAZE_ARRAY), len(DEFAULT_MAZE_ARRAY[0]))

# environment will run for a max of 50 frames before done
FRAME_LIMIT = 50


class MazeEnv(gym.Env):
    """Custom maze environment"""

    # not sure what metadata is
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MazeEnv, self).__init__()

        self.current_step = 0

        # initialize reward range
        self.reward_range = (0, COMPLETION_REWARD + STEP_REWARD + POINT_REWARD)

        # set action space (move up, right, left, down, or don't move)
        self.action_space = spaces.Discrete(5)

        # set observation space (value of every cell is passed to agent)
        self.observation_space = spaces.Box(
            low=np.array([0 for element in range(BOARD_RESOLUTION[0] * BOARD_RESOLUTION[1])]),
            high=np.array([5 for element in range(BOARD_RESOLUTION[0] * BOARD_RESOLUTION[1])]),
            dtype=np.float16
        )

        # for now, initialize maze array to default array
        self.maze_array = DEFAULT_MAZE_ARRAY

        self.position = [0, 0]

    def reset(self):
        self.maze_array = DEFAULT_MAZE_ARRAY

        return self._next_observation()

    def _next_observation(self):
        return self.maze_array

    def _take_action(self, action):
        reward = STEP_REWARD
        done = False
        target_index = zip(self.position, MOVEMENT_ARRAY[action])
        target_index = [x + y for (x, y) in target_index]

        if target_index[0] >= BOARD_RESOLUTION[0] or target_index[1] >= BOARD_RESOLUTION[1] or target_index[0] < 0\
                or target_index[1] < 0 or self.maze_array[target_index[0]][target_index[1]] == 1:
            return reward, False
        elif self.maze_array[target_index[0]][target_index[1]] == 3:
            reward += COMPLETION_REWARD
            done = True
        elif self.maze_array[target_index[0]][target_index[1]] == 2:
            reward += POINT_REWARD

        self.maze_array[self.position[0]][self.position[1]] = 0
        self.maze_array[target_index[0]][target_index[1]] = 4
        self.position = target_index

        # if self.maze_array[target_index[0]][target_index[1]] == 3:
        #     reward += COMPLETION_REWARD
        #     done = True
        # else:
        #     if self.maze_array[target_index[0]][target_index[1]] == 2:
        #         reward += POINT_REWARD
        #     elif self.maze_array[target_index[0]][target_index[1]] == 4:
        #         print("There is already a 4 at {}".format(target_index))
        #     if self.maze_array[target_index[0]][target_index[1]] != 1:
        #         self.maze_array[self.position[0]][self.position[1]] = 0
        #         self.maze_array[target_index[0]][target_index[1]] = 4
        #         self.position = target_index

        return reward, done

    def step(self, action):
        # take a step in environment (with action method defined elsewhere within mazeenv.py)
        reward, done = self._take_action(action)
        self.current_step += 1

        # observe environment states
        obs = self._next_observation()

        # is the environment done yet?
        if not done:
            done = self.current_step >= FRAME_LIMIT

        return obs, reward, done

    def render(self, mode='human', close=False):
        # render environment (for now, just text interface)
        for row in range(BOARD_RESOLUTION[0]):
            for col in range(BOARD_RESOLUTION[1]):
                print(self.maze_array[row][col], end=" ")
            print("\n")


done_list = []

env = MazeEnv()

env.render()
print("\n\n")

obs, reward, done = env.step(1)
done_list.append(done)

env.render()
print("\n\n")

obs, reward, done = env.step(4)
done_list.append(done)

env.render()
print("\n\n")

obs, reward, done = env.step(2)
done_list.append(done)

env.render()
print("\n\n")

obs, reward, done = env.step(3)
done_list.append(done)

env.render()
print("\n\n")

print(done_list)