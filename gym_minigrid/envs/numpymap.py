from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from pathlib import Path
import numpy as np
from gym_minigrid.index_mapping import malmo_object_to_index, minigrid_index_mapping

RESOURCES_DIR = (Path(__file__).parent / './resources').resolve()


class NumpyMap(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.

    * reward_structure is per toggle, 
    faster traige than minecraft 
    if a person toggles the yellow victim somewhat 
    and then leaves and then comes back to it.
    Minecraft resets the toggle count perhaps? YES~ (which is not realistic)

    """

    def __init__(self, index_mapping=minigrid_index_mapping,
                 numpy_array=np.load(Path(RESOURCES_DIR, 'raw_map.npy')),
                 agent_pos=None, agent_dir=None, max_steps=10000, max_goals=26,
                 mission_statement="Triage the victims",
                 reward_structure={
                     'yellow': 25, 'inv_indianyellow': 25,
                     'green': 10, 'inv_green2': 10,
                     'red': 0, 'white': 0},
                 is_goal_needed=False, goal_pos=None):

        self.numpy_array = numpy_array
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self._agent_default_dir = agent_dir
        self._mission_statement = mission_statement
        self.reward_structure = reward_structure
        self.is_goal_needed = is_goal_needed
        self.index_mapping = index_mapping
        self.full_state, self.partial_state = self.encode_initial_states(numpy_array)
        width = numpy_array.shape[1] + 2  # added 2 for the surrounding wall
        height = numpy_array.shape[0] + 2  # added 2 for the surrounding wall

        super().__init__(width=width, height=height, max_steps=max_steps)

    def encode_initial_states(self, numpy_map):
        '''
        Encode the numpy map to get the value in the range of [0, 10] for normalizing
        '''
        [rows, cols] = numpy_map.shape
        full_map = np.zeros((rows + 1, cols), dtype='int')
        partial_map = np.zeros((rows + 1, cols), dtype='int') + 10

        for i in range(rows):
            for j in range(cols):
                if numpy_map[i, j] == 1:  # empty
                    full_map[i, j] = 1
                elif numpy_map[i, j] == 2:  # 2: closed door 3: opened door
                    full_map[i, j] = partial_map[i, j] = 2
                elif numpy_map[i, j] == 4:  # wall
                    full_map[i, j] = partial_map[i, j] = 4
                elif numpy_map[i, j] == 5:  # light switch
                    full_map[i, j] = partial_map[i, j] = 5
                elif numpy_map[i, j] == 255 or numpy_map[i, j] == 11:  # box
                    full_map[i, j] = 6
                elif numpy_map[i, j] == 81:  # 7: yellow victim 8: red victim
                    full_map[i, j] = 7
                elif numpy_map[i, j] == 82:  # green victim
                    full_map[i, j] = 9
                else:  # unknown or unseen
                    full_map[i, j] = partial_map[i, j] = 10
        pos_x, pos_z = self._agent_default_pos[1] - 1, self._agent_default_pos[0] - 1
        full_map[pos_x, pos_z] = partial_map[pos_x, pos_z] = 0  # agent's initial position
        full_map[-1, 0] = partial_map[-1, 0] = self._agent_default_dir  # agent's initial direction
        fwd_pos = self._agent_default_pos + DIR_TO_VEC[self._agent_default_dir]
        fwd_cell = full_map[fwd_pos[1] - 1, fwd_pos[0] - 1]
        full_map[-1, 1] = partial_map[-1, 1] = fwd_cell  # forward object

        return full_map, partial_map

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create the grid
        if isinstance(self.numpy_array, str):
            self.array = np.load(self.numpy_array)
        else:
            self.array = self.numpy_array

        for mc_i in range(0, self.array.shape[0]):
            for mc_j in range(0, self.array.shape[1]):

                # To accomodate any coordinate shift???
                mg_i, mg_j = mc_i + 1, mc_j + 1
                entity_index = int(self.array[mc_i][mc_j])

                entity_name = self.index_mapping['object_mapping'].get(entity_index, None)
                if entity_name is None:
                    print("Check minigrid_index_mapping/object_mapping \
                        in utils/index_mapping.py \
                        as the entity_index {} is not found \
                        in object_mapping".format(entity_index))
                    raise KeyError

                entity_color = self.index_mapping['color_mapping'].get(entity_index, None)
                entity_toggletime = self.index_mapping['toggletimes_mapping'].get(entity_index, None)

                if entity_name in ['agent', 'empty', 'unseen']:
                    continue
                # elif entity_name == 'unseen':
                # self.put_obj(Wall(), mg_j, mg_i)
                else:
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print(entity_index, entity_name, entity_color, entity_toggletime)
                            if entity_toggletime is not None:
                                self.put_obj(entity_class(
                                    color=entity_color,
                                    toggletimes=entity_toggletime), mg_j, mg_i)

                            elif entity_color is not None:
                                self.put_obj(entity_class(
                                    color=entity_color), mg_j, mg_i)
                            else:
                                self.put_obj(entity_class(), mg_j, mg_i)

        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            if self._agent_default_dir is not None:
                self.agent_dir = self._agent_default_dir
            else:
                self.agent_dir = self._rand_int(0, 4)  # assuming random start direction

        else:
            self.place_agent()

        if self.is_goal_needed:
            if self._goal_default_pos is not None:
                goal = Goal()
                self.put_obj(goal, *self._goal_default_pos)
                goal.init_pos, goal.cur_pos = self._goal_default_pos
            else:
                self.place_obj(Goal())

        self.mission = self._mission_statement  # 'Reach the goal'

    def step(self, action):
        """
        Modified step function than regular minigrid 
        that suits Minecraft USAR
        * Change goal reward structure to depend on color
        * Disabled done on facing the goal, done only on 
        max steps or when player leaves the Minecraft world
        TODO:
        * Enable sufficient toggle requirement to activate 
        the reward for pure minimap trajectories collection.
        """
        # obs, reward, done, info = MiniGridEnv.step(self, action)
        reward = 0
        done = False
        info = {}
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        # print(fwd_cell.type)
        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Toggle/activate an object
        elif action == self.actions.toggle:
            # print(fwd_cell.type)
            # print(fwd_cell.is_open)
            if fwd_cell:
                changed = fwd_cell.toggle(self, fwd_pos)
                if changed and fwd_cell.type == 'goal':
                    reward = self.reward_structure[fwd_cell.color]
                    if self.reward_structure[fwd_cell.color] > 0:
                        self.put_obj(Goal('white', toggletimes=0), *fwd_pos)

            # if fwd_cell != None and fwd_cell.type == 'goal':
            #    reward = self.reward_structure[fwd_cell.color]

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                pass
                # print("player near victim")
                # reward = self.reward_structure[fwd_cell.color]
                # done = True
                # reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        self.step_count += 1

        return obs, reward, done, info

    def reset_map(self, numpy_array):
        self.step_count = 0
        self.numpy_array = numpy_array
        obs = MiniGridEnv.reset()
        return obs

    def encode_numpy_map(self):
        return


register(
    id='MiniGrid-NumpyMap-v0',
    entry_point='gym_minigrid.envs:NumpyMap'
)
