import math
import operator
from copy import deepcopy
from functools import reduce
from queue import deque
from enum import IntEnum
from gym_minigrid.minigrid import Goal
from gym_minigrid.window import Window
import collections

import numpy as np
import gym
from gym import error, spaces, utils
from .index_mapping import *
import copy
import torch


class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class DACWrapper(gym.core.Wrapper):
    '''
    Wrapper to zero out the env when episode ends
    '''

    def __init__(self, env):
        super().__init__(env)
        self.env_done = False
        self.last_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.env_done = False
        self.last_obs = obs
        if isinstance(obs, dict):
            self.last_obs = dict()
            for k, v in obs.items():
                self.last_obs[k] = deepcopy(v)
            self.last_obs['image'] = obs['image'] * 0 + 1
        else:
            self.last_obs = obs * 0 + 1
        self.count = 0
        return obs

    def step(self, action):
        self.count += 1
        if self.env_done:
            # Done if time out
            if self.count >= self.env.max_steps:
                return self.last_obs, 0, True, {}
            else:
                return self.last_obs, 0, False, {}
        else:
            # Env is not done, go on
            obs, rew, done, info = self.env.step(action)
            if not done:
                return obs, rew, done, info
            else:
                # Done
                obs = self.last_obs
                self.env_done = True
                if self.count >= self.env.max_steps:
                    return obs, rew, True, info
                else:
                    return obs, rew, False, info

    def render(self, *args, **kwargs):
        img = self.env.render(*args, **kwargs)
        if self.env_done:
            img = img * 0
        return img


class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']


class AgentExtraInfoWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'image': env.observation_space.spaces['image'],
            'pos': gym.spaces.Box(-1, 10000, shape=(2,)),
            'dir': gym.spaces.Box(0, 5, shape=()),
        })

    def observation(self, obs):
        obss = {
            'pos': self.env.agent_pos,
            'dir': self.env.agent_dir,
        }
        for k, v in obs.items():
            obss[k] = v
        return obss

    def get_map(self):
        grid = self.env.grid.encode()
        grid = grid[:, :, 0]
        return grid

    def get_full_map(self):
        env = self.env
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        return full_grid.astype(np.uint8)


class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype='uint8'
        )

    def observation(self, obs):
        img = obs['image']
        out = np.zeros(self.observation_space.shape, dtype='uint8')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {
            'mission': obs['mission'],
            'image': out
        }


class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as the only observation output
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img_partial = env.get_obs_render(
            obs['image'],
            tile_size=self.tile_size
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img_partial
        }


class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return {
            'mission': obs['mission'],
            'image': full_grid
        }


class FullyObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Convert the fully observed wrapper into a one hot tensor
    """

    def __init__(self, env, drop_color=False, keep_classes=None, flatten=True):
        # assert 'FullyObsWrapper' in env.__class__.__name__
        super().__init__(env)
        # Number of classes
        if not keep_classes:
            keep_classes = list(OBJECT_TO_IDX.keys())
        keep_classes.sort(key=lambda x: OBJECT_TO_IDX[x])
        # Save number of classes and find new mapping
        self.num_classes = len(keep_classes)
        # Keep a mapping from old to new mapping so that it becomes easier to map
        # to one hot
        self.object_to_new_idx = dict()
        for idx, k in enumerate(keep_classes):
            self.object_to_new_idx[OBJECT_TO_IDX[k]] = idx

        # Number of colors
        if drop_color:
            self.num_colors = 0
        else:
            self.num_colors = len(COLOR_TO_IDX)
        self.num_states = 4

        self.N = self.num_classes + self.num_colors + self.num_states

        # Define shape of the new environment
        selfenvobs = self.env.observation_space
        try:
            selfenvobs = selfenvobs['image'].shape
        except:
            selfenvobs = selfenvobs.shape
        self.obsshape = list(selfenvobs[:2])
        self.flatten = flatten
        if flatten:
            self.obsshape = np.prod(self.obsshape)
            shape = (self.obsshape * self.N,)
        else:
            shape = tuple(self.obsshape + [self.N])
            self.obsshape = np.prod(self.obsshape)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype='uint8',
        )

    def observation(self, obs):
        # obs = obs.reshape(-1)
        # Get one hot vector
        onehotclass = np.zeros((self.obsshape, self.num_classes), dtype=np.uint8)
        onehotcolor = np.zeros((self.obsshape, self.num_colors), dtype=np.uint8)
        onehotstate = np.zeros((self.obsshape, self.num_states), dtype=np.uint8)
        rangeobs = np.arange(self.obsshape)

        classes = obs[:, :, 0].reshape(-1)
        classes = np.vectorize(self.object_to_new_idx.__getitem__)(classes)
        onehotclass[rangeobs, classes] = 1

        # Go for color
        if self.num_colors > 0:
            colors = obs[:, :, 1].reshape(-1)
            onehotcolor[rangeobs, colors] = 1

        states = obs[:, :, 2].reshape(-1)
        onehotstate[rangeobs, states] = 1

        # Concat along the number of states dimension
        onehotobs = np.concatenate([onehotclass, onehotcolor, onehotstate], 1)
        if self.flatten:
            return onehotobs.reshape(-1)
        else:
            return onehotobs.reshape(self.observation_space.shape)


class AppendActionWrapper(gym.core.Wrapper):
    """
    Append the previous actions taken
    """

    def __init__(self, env, K):
        super().__init__(env)
        # K is the number of actions (including present)
        # size is the number of one hot vector
        self.K = K
        self.actsize = env.action_space.n
        self.history = deque([np.zeros(self.actsize) for _ in range(self.K)])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.observation_space.shape[0] + self.actsize * self.K,),
            dtype='uint8'
        )

    def reset(self, **kwargs):
        self.history = deque([np.zeros(self.actsize) for _ in range(self.K)])
        obs = self.env.reset(**kwargs)
        actall = np.concatenate(self.history)
        actall = actall.astype(np.uint8)
        # Append it to obs
        obs = np.concatenate([obs, actall])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # get one hot action
        act = np.zeros((self.actsize))
        act[action] = 1
        # update history
        self.history.popleft()
        self.history.append(act)
        actall = np.concatenate(self.history)
        actall = actall.astype(np.uint8)
        # Append it to obs
        obs = np.concatenate([obs, actall])
        return obs, reward, done, info


class GoalPolicyWrapper(gym.core.GoalEnv):
    """
    Encode a goal policy based on whether the agent reached the goal or not
    This is for simple navigation based goals only
    """

    def __init__(self, env, ):
        self.env = env
        assert isinstance(self.env, FullyObsOneHotWrapper)
        self.observation_space = gym.spaces.Dict({
            'observation': env.observation_space,
            'achieved_goal': env.observation_space,
            'desired_goal': env.observation_space,
        })
        self.action_space = env.action_space

    def _get_goals(self, Obs):
        # Create achieved and desired goals
        agentidx = self.env.object_to_new_idx[OBJECT_TO_IDX['agent']]
        emptyidx = self.env.object_to_new_idx[OBJECT_TO_IDX['empty']]
        goalidx = self.env.object_to_new_idx[OBJECT_TO_IDX['goal']]
        # Init the goals
        obs = Obs.reshape(self.env.obsshape, -1)
        achieved = obs + 0
        desired = obs + 0
        # For achieved, just erase the goal
        achieved[:, goalidx] = 0
        # For desired, find the goal and replace by agent.
        # Replace the agent with empty
        agent_pos = np.where(desired[:, agentidx] > 0)
        goal_pos = np.where(desired[:, goalidx] > 0)

        desired[agent_pos, agentidx] = 0
        desired[agent_pos, emptyidx] = 1

        desired[goal_pos, goalidx] = 0
        desired[goal_pos, agentidx] = 1
        return achieved.reshape(-1), desired.reshape(-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        env = self.env
        while True:
            if hasattr(env, '_reward'):
                return env._reward()
            else:
                env = env.env

    def reset(self, ):
        obs = self.env.reset()
        achieved, desired = self._get_goals(obs)

        return {
            'observation': obs,
            'achieved_goal': achieved,
            'desired_goal': desired,
        }

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        achieved, desired = self._get_goals(obs)
        obs_new = {
            'observation': obs,
            'achieved_goal': achieved,
            'desired_goal': desired,
        }
        return obs_new, rew, done, info


class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class InvertColorsWrapper(gym.core.Wrapper):
    """
    Invert colors
    """

    def __init__(self, env):
        self.env = env
        super().__init__(env)

    def render(self, mode='rgb_array', tile_size=8,
               alpha_visiblity=0.3, alpha_unseen=0.9):
        """
        Render the whole-grid human view
        """
        # Render the whole grid
        mgimg = self.env.unwrapped.grid.render(
            tile_size,
            self.env.unwrapped.agent_pos,
            self.env.unwrapped.agent_dir,
            highlight_mask=None,
            prev_pos_mask=None,
        )
        mgimg = 255 - mgimg
        return mgimg


class HumanFOVWrapper(gym.core.Wrapper):
    """
    Wrapper to produce human like FOV where angle of view is 51
    Returns the observation of the size of the actual map
    """

    def __init__(self, env, agent_pos=None, frame_stack=3):
        super().__init__(env)

        self.env = env
        self.initial_agent_pos = agent_pos
        self.frame_stack = frame_stack

        self.window = Window('gym_minigrid - Falcon ' + str(env.difficulty))

        class MCActions(IntEnum):
            # moves
            left = 0
            right = 1
            forward = 2
            backward = 3
            act_triage = 4
            no_act = 5

        from pathlib import Path
        import csv

        RESOURCES_DIR = (Path(__file__).parent / './envs/resources').resolve()
        trigger_areas_cvs_path = Path(RESOURCES_DIR, 'MapInfo.csv')

        self.trigger_areas_dict = {}
        with open(trigger_areas_cvs_path)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                if row[0] != 'LocationXYZ':
                    self.trigger_areas_dict[(int(row[0].split(" ")[0]) + 2111, int(row[0].split(" ")[-1]) - 141)] = row[
                        -1]

        self.areas = {'Outside of Building': [[1, 2], [14, 36]], 'Entrance': [[16, 3], [21, 8]],
                      'Security Office': [[23, 3], [25, 9]],
                      'Open Break Area': [[27, 3], [34, 9]], 'Executive Suite 1': [[36, 3], [51, 9]],
                      'Executive Suite 2': [[53, 3], [66, 9]], 'King Chris\'s Office 1': [[68, 3], [74, 9]],
                      'King Chris\'s Office 2': [[75, 3], [82, 17]], 'The King\'s Terrace': [[84, 3], [91, 17]],
                      'Room 101': [[75, 20], [82, 27]], 'Room 102': [[75, 29], [82, 36]],
                      'Room 103': [[75, 43], [82, 50]],
                      'Room 104': [[66, 43], [73, 50]], 'Room 105': [[57, 43], [64, 50]],
                      'Room 106': [[48, 43], [55, 50]], 'Room 107': [[39, 43], [46, 50]],
                      'Room 108': [[30, 43], [37, 50]],
                      'Room 109': [[21, 43], [28, 50]], 'Room 110': [[12, 43], [19, 50]],
                      'Room 111': [[3, 43], [10, 50]],
                      'The Computer Farm': [[16, 16], [32, 36]], 'Janitor': [[40, 16], [44, 18]],
                      'Men\'s Room': [[39, 20], [44, 27]],
                      'Women\'s Room': [[39, 29], [44, 36]],
                      'Amway Conference Room': [[47, 16], [54, 25]], 'Mary Kay Conference Room': [[47, 27], [54, 36]],
                      'Herbalife Conference Room': [[61, 16], [68, 36]]}

        self.actions = MCActions
        self.action_idx_mapping = [MCActions.left, MCActions.right, MCActions.forward, MCActions.backward,
                                   MCActions.act_triage, MCActions.no_act]

        self.action_time_cost_list = {'up': 0.15,
                                      'down': 0.15,
                                      'left': 0.15,
                                      'right': 0.15,
                                      'act_triage': 7.5,
                                      'no_act': 1}

        self.time_left = env.total_game_duration
        self.yellow_victim_lifetime = env.yellow_victim_lifetime
        self.is_yellow_victim_alive = True
        self.yellow_victim_colors = {'yellow', 'indianyellow', 'inv_indianyellow'}

        self.step_count = 0
        self.max_steps = 100000
        # Override the agent_view_size, make sure it is not used
        self.width = env.width
        self.height = env.height

        env.unwrapped.agent_view_size = None  # (env.width, env.height)
        self.grid = copy.deepcopy(env.unwrapped.grid)
        self.initial_grid = copy.deepcopy(env.unwrapped.grid)
        self.agent_pos = copy.deepcopy(self.env.unwrapped.agent_pos)
        self.initial_agent_pos = copy.deepcopy(self.env.unwrapped.agent_pos)

        self.gt_map_state = copy.deepcopy(self.env.initial_gt_map_state)
        self.initial_gt_map_state = copy.deepcopy(self.env.initial_gt_map_state)
        self.observed_map_state = copy.deepcopy(self.env.gt_map_state)

        self.initial_victim_status = []
        self.victim_locations = []
        for w in range(self.width):
            for h in range(self.height):
                if self.env.initial_gt_map_state[h][w] == 81:
                    self.initial_victim_status.append(1)
                    self.victim_locations.append([w, h])
                elif self.env.initial_gt_map_state[h, w] == 82:
                    self.initial_victim_status.append(2)
                    self.victim_locations.append([w, h])

        self.initial_victim_status = np.array(self.initial_victim_status)
        self.victim_locations = np.array(self.victim_locations)
        self.victim_status = copy.deepcopy(self.initial_victim_status)

        self.prev_pos_mask = -np.ones((self.width, self.height))  # , dtype=bool)

        # Compute observation space
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(43,),
            dtype='uint8'
        )

        # rl-zoo doesn't support dictionary
        # self.observation_space = gym.spaces.Dict({
        #     'image': observation_space
        # })

        self.observation_space = observation_space
        self.obs = np.zeros((1, self.height, self.width))
        self.sim_state = np.zeros((43,), dtype='uint8')
        self.pre_location = collections.deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.pre_location.append(self.agent_pos)

        # SIMPLE: Action is either move forward, turn left by 10 degrees, turn right by 10 degrees, toggle
        # FUTURE TODO: Action consists of moving to a neighbour cells by 1 or 2 blocks (24 options)
        # and turning by 15 degrees (24 options)
        self.action_space = gym.spaces.Discrete(6)

        self.total_goals = 24 + 10
        self.saved_yellow_victim = 0
        self.saved_green_victim = 0
        self.goals_acheived = 0

        self.triaging_time = 0

        self.visible_grid = np.zeros((env.height, env.width))
        self.observed_absolute_map = np.zeros((env.height, env.width))

        current_pos = self.agent_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell != None:
            if current_cell.type == 'goal':
                if current_cell.color == 'inv_indianyellow' or current_cell.color == 'yellow':
                    self.current_obj = 50
                elif current_cell.color == 'inv_green2' or current_cell.color == 'green':
                    self.current_obj = 10
                else:
                    self.current_obj = 0
        else:
            self.current_obj = 0

    def inview2D_with_opaque_objects(self, yaw, distance_resolution_factor=1, angle_resolution_factor=1):
        # envsize should be same as the grid size.

        # Assuming that the player has a headlight with them
        # This function is independent of environment's visibility
        theta_mu = np.pi * yaw / 180.
        theta_sigma = 0.9
        radius_mu = 0.01
        radius_sigma = 10.

        angle_resolution = self.angle * angle_resolution_factor
        dist_resolution = self.distance * distance_resolution_factor
        binary_visible_grid_mask = np.zeros((self.grid.height, self.grid.width), dtype=np.uint8)
        visible_prob_grid = np.zeros((self.grid.height, self.grid.width), dtype=np.float64)

        thetas_in_deg_array = np.linspace(yaw - self.angle, yaw + self.angle, angle_resolution)
        theta_in_rad_array = np.pi * thetas_in_deg_array / 180.
        radius_array = np.linspace(1., self.distance, dist_resolution)

        radius, theta = np.meshgrid(radius_array, theta_in_rad_array, sparse=True)

        p_theta = np.exp(-((theta - theta_mu) ** 2 / (2.0 * theta_sigma ** 2)))
        p_radius = np.exp(-((radius - radius_mu) ** 2 / (2.0 * radius_sigma ** 2)))
        p_total = p_radius * p_theta

        visible_prob_grid[int(round(self.agent_pos[1]))][int(round(self.agent_pos[0]))] = 1.
        coord_z = radius * np.cos(theta) + self.agent_pos[1]  # zpos - self.origin_coord['z']
        coord_x = -radius * np.sin(theta) + self.agent_pos[0]  # xpos - self.origin_coord['x']

        for i in range(angle_resolution):
            for j in range(dist_resolution):
                index_z = int(round(coord_z[i][j]))
                index_x = int(round(coord_x[i][j]))
                if index_z >= 0 and index_z < self.grid.height and index_x >= 0 and index_x < self.grid.width:
                    if not visible_prob_grid[index_z][index_x]:
                        item = self.grid.grid[index_z * self.grid.width + index_x]
                        if item is not None and not item.see_behind():
                            binary_visible_grid_mask[index_z][index_x] = 1
                            break
                        else:
                            visible_prob_grid[index_z][index_x] = p_total[i][j]
                            binary_visible_grid_mask[index_z][index_x] = 1

        visible_prob_grid[int(round(self.agent_pos[1]))][int(round(self.agent_pos[0]))] = 1.
        binary_visible_grid_mask[int(round(self.agent_pos[1]))][int(round(self.agent_pos[0]))] = 1.

        self.visible_grid = binary_visible_grid_mask

        return visible_prob_grid

    def victim_detection(self, pos_x, pos_z):
        if (pos_x, pos_z) in self.trigger_areas_dict.keys():
            trigger_area = self.trigger_areas_dict[(pos_x, pos_z)]
            if trigger_area in ['Herbalife Conference Room North Door', 'Herbalife Conference Room South Door']:
                trigger_area = 'Herbalife Conference Room'
            return self.search_current_area(trigger_area)
        return 0

    def gen_obs(self):
        self.pre_location.append(self.agent_pos)
        for i in range(self.frame_stack):
            self.obs[0, -1, 2 * i] = self.pre_location[i][0]
            self.obs[0, -1, 2 * i + 1] = self.pre_location[i][1]

        self.sim_state[0:len(self.victim_status)] = self.victim_status
        for i in range(self.frame_stack):
            self.sim_state[len(self.victim_status) + 2 * i] = self.pre_location[i][0]
            self.sim_state[len(self.victim_status) + 2 * i + 1] = self.pre_location[i][1]

        self.sim_state[len(self.victim_status) + self.frame_stack * 2] = int(self.is_yellow_victim_alive)

        if len(np.where(np.all(self.victim_locations == self.agent_pos, axis=1))[0]) > 0:
            self.sim_state[-2] = self.victim_status[int(np.where(
                np.all(self.victim_locations == self.agent_pos, axis=1))[0])]
        else:
            self.sim_state[-2] = 0
        self.sim_state[-1] = self.triaging_time

        return copy.deepcopy((self.sim_state).astype('uint8'))

    def check_remaining_goals(self):
        if self.goals_acheived == self.total_goals:
            return True
        return False

    def _reward(self, fwd_cell):
        if fwd_cell.prevcolor == 'yellow' or fwd_cell.prevcolor == 'inv_indianyellow':
            self.goals_acheived += 1
            self.saved_yellow_victim += 1
            return 30
        elif fwd_cell.prevcolor == 'green' or fwd_cell.prevcolor == 'inv_green2':
            self.goals_acheived += 1
            self.saved_green_victim += 1
            return 10
        else:
            return 0

    def reset(self, **kwargs):
        self.time_left = self.env.total_game_duration
        self.is_yellow_victim_alive = True

        self.step_count = 0
        self.max_steps = 100000

        self.width = self.env.width
        self.height = self.env.height

        self.env.unwrapped.agent_view_size = None  # (env.width, env.height)
        # self.grid = self.env.unwrapped.grid
        self.grid = copy.deepcopy(self.initial_grid)
        # self.grid = self.initial_grid
        # self.agent_pos = self.env.unwrapped.agent_pos
        self.agent_pos = copy.deepcopy(self.initial_agent_pos)
        self.prev_pos_mask = -np.ones((self.width, self.height))  # , dtype=bool)

        self.total_goals = 24 + 10
        self.saved_yellow_victim = 0
        self.saved_green_victim = 0
        self.goals_acheived = 0

        self.triaging_time = 0

        self.visible_grid = np.zeros((self.env.height, self.env.width))
        self.observed_absolute_map = np.zeros((self.env.height, self.env.width))

        self.gt_map_state = copy.deepcopy(self.initial_gt_map_state)
        self.observed_map_state = copy.deepcopy(self.initial_gt_map_state)

        for _ in range(self.frame_stack):
            self.pre_location.append(self.agent_pos)

        current_pos = self.agent_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell != None:
            if current_cell.type == 'goal':
                if current_cell.color == 'inv_indianyellow' or current_cell.color == 'yellow':
                    self.current_obj = 50
                elif current_cell.color == 'inv_green2' or current_cell.color == 'green':
                    self.current_obj = 10
                else:
                    self.current_obj = 0
        else:
            self.current_obj = 0

        self.observed_map_state[
            self.agent_pos[1], self.agent_pos[0]] = 100 + self.current_obj + self.triaging_time + self.victim_detection(
            self.agent_pos[0], self.agent_pos[1])
        # Return first observation
        self.obs = np.zeros((1, self.height, self.width))
        self.victim_status = copy.deepcopy(self.initial_victim_status)
        self.sim_state = np.zeros((43,), dtype='uint8')

        obs = self.gen_obs()
        # for _ in range(self.frame_stack):
        #     obs = self.gen_obs()
        return obs

    def get_legal_actions(self):
        return np.array([0, 1, 2, 3, 4, 5])

    def get_current_area(self):
        pos_x, pos_z = self.agent_pos
        current_area = 'Hallway'
        for name, locations in self.areas.items():
            try:
                if locations[0][0] <= pos_x <= locations[1][0] and locations[0][1] <= pos_z <= locations[1][1]:
                    current_area = name
                    break
            except:
                print()
        return current_area

    def search_current_area(self, current_area):
        if current_area == 'King Chris\'s Office':
            current_areas = ['King Chris\'s Office 1', 'King Chris\'s Office 2']
        elif current_area in ['Herbalife Conference Room North Door', 'Herbalife Conference Room South Door']:
            current_areas = ['Herbalife Conference Room']
        else:
            current_areas = [current_area]

        beep_times = 0
        for area in current_areas:
            tl_x, tl_z = self.areas[area][0][0], self.areas[area][0][1]
            br_x, br_z = self.areas[area][1][0], self.areas[area][1][1]

            for x in range(tl_x, br_x + 1):
                for z in range(tl_z, br_z + 1):
                    if self.grid.grid[z * self.width + x] is not None:
                        if self.grid.grid[z * self.width + x].type == 'goal':
                            if self.grid.grid[z * self.width + x].color == 'yellow' or self.grid.grid[
                                z * self.width + x].color == 'inv_indianyellow':
                                return 2
                            elif self.grid.grid[z * self.width + x].color == 'green' or self.grid.grid[
                                z * self.width + x].color == 'inv_green2':
                                beep_times = 1

        return beep_times

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def step(self, action):
        action = self.action_idx_mapping[action]
        self.step_count += 1

        reward = 0
        done = False

        # Swap yellow victims with red victims
        if (
                self.env.total_game_duration - self.time_left) >= self.yellow_victim_lifetime and self.is_yellow_victim_alive:
            for i in range(self.width):
                for j in range(self.height):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type == 'goal' and cell.color in self.yellow_victim_colors:
                        self.put_obj(Goal('red', toggletimes=math.inf), i, j)
                        # self.gt_map_state[j, i] = 80
                        # change to 83 to decrease the observation space
                        self.gt_map_state[j, i] = 83
                        self.victim_status[self.victim_status == 1] = 0
            self.is_yellow_victim_alive = False

        # Get the position of the agent
        current_pos = self.agent_pos
        # Get the position up upon the agent
        up_pos = self.agent_pos + DIR_TO_8_VEC[4]
        # Get the position under the agent
        down_pos = self.agent_pos + DIR_TO_8_VEC[0]
        # Get the position on the left of the agent
        left_pos = self.agent_pos + DIR_TO_8_VEC[2]
        # Get the position on the right of the agent
        right_pos = self.agent_pos + DIR_TO_8_VEC[6]

        # Get the contents of the cell in front of the agent
        current_cell = self.grid.get(*current_pos)
        up_cell = self.grid.get(*up_pos)
        # Get the contents of the cell behind the agent
        down_cell = self.grid.get(*down_pos)
        left_cell = self.grid.get(*left_pos)
        right_cell = self.grid.get(*right_pos)

        #  Move left
        if action == self.actions.left:
            if left_cell == None or left_cell.can_overlap():
                self.agent_pos = left_pos
                if left_cell != None and left_cell.type == 'box':
                    self.grid.is_on_box(True)
                else:
                    self.grid.is_on_box(False)
            self.time_left -= self.action_time_cost_list['left']
        # Move right
        elif action == self.actions.right:
            if right_cell == None or right_cell.can_overlap():
                self.agent_pos = right_pos
                if right_cell != None and right_cell.type == 'box':
                    self.grid.is_on_box(True)
                else:
                    self.grid.is_on_box(False)
            self.time_left -= self.action_time_cost_list['right']
        # Move up
        elif action == self.actions.forward:
            if up_cell == None or up_cell.can_overlap():
                self.agent_pos = up_pos
                if up_cell != None and up_cell.type == 'box':
                    self.grid.is_on_box(True)
                else:
                    self.grid.is_on_box(False)
            self.time_left -= self.action_time_cost_list['up']
        # Move down
        elif action == self.actions.backward:
            if down_cell == None or down_cell.can_overlap():
                self.agent_pos = down_pos
                if down_cell != None and down_cell.type == 'box':
                    self.grid.is_on_box(True)
                else:
                    self.grid.is_on_box(False)
            self.time_left -= self.action_time_cost_list['down']

        # Triage a victim
        elif action == self.actions.act_triage:
            if current_cell != None:
                if current_cell.type == 'goal' and current_cell.color != current_cell.triage_color:
                    if current_cell.required_toggle_times == current_cell.toggletimes:
                        self.triaging_victim = current_cell
                        current_cell.act_triage(self, current_pos)
                        self.triaging_time += 1
                    elif current_cell == self.triaging_victim:
                        current_cell.act_triage(self, current_pos)
                        self.triaging_time += 1
                    else:
                        current_cell.toggletimes = current_cell.required_toggle_times
                        self.triaging_victim = current_cell
                        current_cell.act_triage(self, current_pos)
                        self.triaging_time += 1

                    if current_cell.toggletimes <= 0:
                        reward = self._reward(current_cell)
                        self.gt_map_state[current_pos[1], current_pos[0]] = 83
                        self.triaging_time = 0

                        if np.where(np.all(self.victim_locations == self.agent_pos, axis=1)):
                            self.victim_status[
                                int(np.where(np.all(self.victim_locations == self.agent_pos, axis=1))[0])] = 0
                    else:
                        reward = 0
            else:
                self.triaging_time = 0
                reward = 0

            self.time_left -= self.action_time_cost_list['act_triage']

        elif action == self.actions.no_act:
            reward = 0
            self.time_left -= self.action_time_cost_list['no_act']
        else:
            assert False, "unknown action"

        if action != self.actions.act_triage:
            self.triaging_time = 0
            self.triaging_victim = None

        if self.step_count >= self.max_steps or self.time_left <= 0:
            done = True

        # Update observed area so far
        self.observed_absolute_map = np.where(self.visible_grid != 0,
                                              self.visible_grid, self.observed_absolute_map)

        # self.env.observed_map_state = np.where(self.visible_grid != 0,
        #                                        self.visible_grid * self.env.gt_map_state, self.env.observed_map_state)
        # update agent's position and direction
        current_pos = self.agent_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell != None:
            if current_cell.type == 'goal':
                if current_cell.color == 'inv_indianyellow' or current_cell.color == 'yellow':
                    self.current_obj = 50
                elif current_cell.color == 'inv_green2' or current_cell.color == 'green':
                    self.current_obj = 10
                else:
                    self.current_obj = 0
        else:
            self.current_obj = 0

        self.observed_map_state = copy.deepcopy(self.gt_map_state)
        # self.observed_map_state[100 <= self.observed_map_state <= 101] = 1
        self.observed_map_state[
            self.agent_pos[1], self.agent_pos[0]] = 100 + self.current_obj + self.triaging_time + self.victim_detection(
            self.agent_pos[0], self.agent_pos[1])
        self.prev_pos_mask[self.agent_pos[0], self.agent_pos[1]] = 270
        obs = self.gen_obs()

        return obs, reward, done, {}

    def preprocess(self, array, visible_granularity, tile_size):
        """scale, n, stack"""

        def scale_by_factor(im, factor):
            """scale the array by given factor"""
            return np.array([[im[int(r / factor)][int(c / factor)]
                              for c in range(len(im[0]) * factor)] for r in range(len(im) * factor)])

        out = scale_by_factor(array, tile_size)
        # out = np.pad(out, pad_width=1*visible_granularity, mode='constant', constant_values=0)
        out = np.stack([out] * 3, -1)
        return out

    def render(self, mode='rgb_array', tile_size=6):
        img = self.render2('rgb_array', tile_size=tile_size)
        self.window.show_img(img)
        return img

    def render2(self, mode='rgb_array', highlight=False, tile_size=8,
                alpha_visiblity=0.3, alpha_unseen=0.6):
        """
        Render the whole-grid human view
        """
        # Render the whole grid
        mgimg = self.grid.render(
            tile_size,
            self.agent_pos,
            agent_dir=270,
            highlight_mask=None,
            prev_pos_mask=self.prev_pos_mask,
        )

        if highlight:

            # breakpoint()
            observed_img = np.where(
                self.preprocess(self.visible_grid, self.visible_granularity, tile_size),
                (mgimg + alpha_visiblity * (255 - mgimg)).clip(0, 255).astype(np.uint8),
                np.where(self.preprocess(self.observed_absolute_map, self.visible_granularity, tile_size),
                         mgimg,
                         (mgimg + alpha_unseen * (255 - mgimg)).clip(0, 255).astype(np.uint8)
                         )
            )
        else:
            observed_img = mgimg
        # invert
        observed_img = 255 - observed_img
        return observed_img

    def renderFoV(self, mode='rgb_array', highlight=True, tile_size=8,
                  alpha_visiblity=0.3, alpha_unseen=0.9):

        # Render the whole grid
        mgimg = self.grid.render(
            tile_size,
            self.agent_pos,
            self.yaw,
            highlight_mask=None,
            prev_pos_mask=self.prev_pos_mask,
        )
        # Mask with FoV
        fov_img = np.where(
            self.preprocess(self.visible_grid, self.visible_granularity, tile_size),
            mgimg,
            # (mgimg + alpha_visiblity*(255-mgimg)).clip(0, 255).astype(np.uint8),
            (mgimg + alpha_unseen * (255 - mgimg)).clip(0, 255).astype(np.uint8))

        # invert
        fov_img = 255 - fov_img
        return fov_img


class IRLRewardWrapper(gym.core.Wrapper):
    """
    Return the IRL reward instead of the reward from original domain.
    """

    def __init__(self, env, discrim, device, alpha=1, add_exploration_bonus=True):
        super().__init__(env)
        self.discrim = discrim
        self.device = device
        self.action_time_cost_list = [0.15, 0.15, 0.15, 0.15, 7.5, 1]
        self.alpha = alpha
        self.add_exploration_bonus = add_exploration_bonus
        self.state_counts = {}

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        state = deepcopy(self.sim_state.astype('uint8'))
        t_state = torch.Tensor(state / 255)
        t_action = torch.Tensor([action / 255])
        state_action = torch.cat([t_state, t_action]).to(self.device)
        self.discrim.eval()
        with torch.no_grad():
            prob = self.discrim(state_action)[0].item()

        if action == 4:
            if state[-2] != 0:
                irl_reward = self.alpha * math.log(prob + 1e-8) / 20
            else:
                irl_reward = self.alpha * math.log(prob + 1e-8) / 20 - (1 - self.alpha) * self.action_time_cost_list[
                    action] / 7.5
        else:
            irl_reward = self.alpha * math.log(prob + 1e-8) / 20 - (1 - self.alpha) * self.action_time_cost_list[
                action] / 7.5

        if self.add_exploration_bonus:
            irl_reward += self.get_exploration_bonus(state, action)

        _, _, done, _ = self.env.step(action)
        return copy.deepcopy(self.sim_state.astype('uint8')), irl_reward, done, {}

    def get_exploration_bonus(self, state, action):
        # Get the count for this key
        pre_count = 0
        state_action = (tuple(state), action)
        if state_action in self.state_counts:
            pre_count = self.state_counts[state_action]

        # Update the count for this key
        new_count = pre_count + 1
        self.state_counts[state_action] = new_count

        # bonus = 1 / math.sqrt(new_count)
        bonus = 1 / new_count
        return bonus
