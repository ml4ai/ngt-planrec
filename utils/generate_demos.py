'''
convert the human subjects data to the actions in minigrid and classify data into different classes
in the file, we ignore unsuccessful triaging and consider it as no action
besides, we ignore conditions to have a larger data set for training
'''

from os import listdir
from os.path import isfile, join
from pathlib import Path
import pickle
import action_parser as action_parser
from action_parser import json_to_action
import gym
from gym_minigrid.wrappers import HumanFOVWrapper
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser(description='generate demonstrations and training and test sets')
parser.add_argument('--frame_stack', type=int, default=3,
                    help='the number of frame stack in one state')
args = parser.parse_args()


def collect_demos(demonstrations, actions, difficulty, episode_reward):
    env = gym.make('MiniGrid-MinimapForFalcon-v0', difficulty=difficulty)
    env = HumanFOVWrapper(env, frame_stack=args.frame_stack)
    obs = env.reset()

    action_num = 0
    epi_states = []
    epi_actions = []
    minigrid_episode_reward = 0

    for action in actions:
        # demo = np.zeros((len(obs)), dtype='uint8')
        # demo = obs
        # demo[-1] = action  # encode (the action + 200) into the last item of observation
        epi_states.append(obs)
        epi_actions.append(action)
        obs, reward, done, _ = env.step(action)
        action_num += 1
        minigrid_episode_reward += reward
        if done:
            env.close()
            break
    # calibration may cause bias for the episode reward, so we can tolerate one yellow victim and one green victim at most.
    # if abs(minigrid_episode_reward - episode_reward) <= 40:
    #     demos.append([demonstration])
    demonstrations['states'].append(epi_states)
    demonstrations['actions'].append(epi_actions)
    demonstrations['rewards'].append(minigrid_episode_reward)

    return demonstrations


def save_files(demos, file_name):
    saved_file_name = '../expert_demos/' + file_name + '.p'
    pickle.dump(demos, open(saved_file_name, "wb"))


def main():
    RESOURCES_DIR = (Path(__file__).parent / '../data').resolve()
    files = [f for f in listdir(RESOURCES_DIR) if isfile(join(RESOURCES_DIR, f))]

    # 9 classes of demonstrations
    e_y_demos = {'states': [], 'actions': [], 'rewards': []}
    e_g_demos = {'states': [], 'actions': [], 'rewards': []}
    e_o_demos = {'states': [], 'actions': [], 'rewards': []}
    m_y_demos = {'states': [], 'actions': [], 'rewards': []}
    m_g_demos = {'states': [], 'actions': [], 'rewards': []}
    m_o_demos = {'states': [], 'actions': [], 'rewards': []}
    d_y_demos = {'states': [], 'actions': [], 'rewards': []}
    d_g_demos = {'states': [], 'actions': [], 'rewards': []}
    d_o_demos = {'states': [], 'actions': [], 'rewards': []}

    e_y_num = 0
    e_g_num = 0
    e_o_num = 0
    m_y_num = 0
    m_g_num = 0
    m_o_num = 0
    d_y_num = 0
    d_g_num = 0
    d_o_num = 0

    for file_name in files:
        if 'HSRData_TrialMessages_CondBtwn' in file_name:
            print('processing ' + file_name)
            actions, invalid_data_found, strategy, episode_reward = json_to_action(file_name)
            if not invalid_data_found:
                if 'FalconEasy' in file_name:
                    if strategy == 'green':
                        e_g_num += 1
                        e_g_demos = collect_demos(e_g_demos, actions, 'easy', episode_reward)
                    elif strategy == 'yellow':
                        e_y_num += 1
                        e_y_demos = collect_demos(e_y_demos, actions, 'easy', episode_reward)
                    else:
                        e_o_num += 1
                        e_o_demos = collect_demos(e_o_demos, actions, 'easy', episode_reward)
                elif 'FalconMed' in file_name:
                    if strategy == 'green':
                        m_g_num += 1
                        m_g_demos = collect_demos(m_g_demos, actions, 'medium', episode_reward)
                    elif strategy == 'yellow':
                        m_y_num += 1
                        m_y_demos = collect_demos(m_y_demos, actions, 'medium', episode_reward)
                    else:
                        m_o_num += 1
                        m_o_demos = collect_demos(m_o_demos, actions, 'medium', episode_reward)
                elif 'FalconHard' in file_name:
                    if strategy == 'green':
                        d_g_num += 1
                        d_g_demos = collect_demos(d_g_demos, actions, 'difficult', episode_reward)
                    elif strategy == 'yellow':
                        d_y_num += 1
                        d_y_demos = collect_demos(d_y_demos, actions, 'difficult', episode_reward)
                    else:
                        d_o_num += 1
                        d_o_demos = collect_demos(d_o_demos, actions, 'difficult', episode_reward)
            else:
                print('failed file: ' + file_name)
                action_parser.invalid_data_found = False

    class_number = np.zeros((3, 3), dtype='uint8')
    print('number for each class: ')
    class_number[0, 0] = e_y_num
    save_files(e_y_demos, 'e_y_demos')
    print('e_y_num: ' + str(e_y_num))
    class_number[0, 1] = e_g_num
    save_files(e_g_demos, 'e_g_demos')
    print('e_g_num: ' + str(e_g_num))
    class_number[0, 2] = e_o_num
    save_files(e_o_demos, 'e_o_demos')
    print('e_o_num: ' + str(e_o_num))

    class_number[1, 0] = m_y_num
    save_files(m_y_demos, 'm_y_demos')
    print('m_y_num: ' + str(m_y_num))
    class_number[1, 1] = m_g_num
    save_files(m_g_demos, 'm_g_demos')
    print('m_g_num: ' + str(m_g_num))
    class_number[1, 2] = m_o_num
    save_files(m_o_demos, 'm_o_demos')
    print('m_o_num: ' + str(m_o_num))

    class_number[2, 0] = d_y_num
    save_files(d_y_demos, 'd_y_demos')
    print('d_y_num: ' + str(d_y_num))
    class_number[2, 1] = d_g_num
    save_files(d_g_demos, 'd_g_demos')
    print('d_g_num: ' + str(d_g_num))
    class_number[2, 2] = d_o_num
    save_files(d_o_demos, 'd_o_demos')
    print('d_o_num: ' + str(d_o_num))

    save_files(class_number, 'class_number')

    print('finished!')


if __name__ == '__main__':
    main()
