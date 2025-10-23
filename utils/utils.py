'''
According to the number of demonstrations and the test set ratio,
we separate the demonstrations into the training and test sets
return training set, test set, observation index list for training set, the number of observations in training set
'''

import numpy as np
import random


def gen_training_and_test_sets(demos, test_set_ratio=0.1):
    training_set = {'states': [], 'actions': [], 'rewards': []}
    test_set = {'states': [], 'actions': [], 'rewards': []}
    training_set_idx_list = []

    demos_len = len(demos['states'])
    if demos_len < 2:
        raise NameError('Should have more than 2 trajectories to train your model!')
        return training_set, test_set
    test_set_num = int(max(np.ceil(demos_len * test_set_ratio), 1))
    test_idx = np.sort(np.random.choice(range(demos_len), test_set_num, replace=False)).tolist()
    starting_idx = 0
    training_idx = 0
    for idx in range(demos_len):
        if idx in test_idx:
            test_set['states'].append(demos['states'][idx])
            test_set['actions'].append(demos['actions'][idx])
            test_set['rewards'].append(demos['rewards'][idx])
        else:
            training_set['states'].append(demos['states'][idx])
            training_set['actions'].append(demos['actions'][idx])
            training_set['rewards'].append(demos['rewards'][idx])
            training_set_idx_list.append([training_idx, starting_idx, starting_idx + len(demos['states'][idx]) - 1])
            starting_idx = starting_idx + len(demos['states'][idx])
            training_idx += 1

    return training_set, test_set, training_set_idx_list, starting_idx


def gen_training_batch(training_set, batch_size, index_list):
    len_obs = index_list[-1][-1] + 1
    obs_index = [i for i in range(len_obs)]
    random.shuffle(obs_index)
    obs_len = len(training_set['states'][0][0])

    pre_states_batches = []
    sub_pre_states_batch = []
    state_batch = []
    sub_state_batch = []
    action_labels = []
    sub_action_labels = []
    reward_labels = []
    sub_reward_labels = []
    num_states = []
    sub_num_states = []

    for oi in obs_index:
        for il in index_list:
            if il[1] <= oi <= il[2]:
                if oi - il[1] == 0:
                    sa_pairs = np.zeros((1, obs_len + 1))
                    sub_pre_states_batch.append(sa_pairs)
                    sub_state_batch.append(training_set['states'][il[0]][0])
                    sub_action_labels.append(training_set['actions'][il[0]][0])
                    sub_reward_labels.append(training_set['rewards'][il[0]])
                    sub_num_states.append(1)
                else:
                    start_idx = 0
                    end_idx = oi - il[1]
                    states = np.array(training_set['states'][il[0]][start_idx:end_idx])
                    actions = np.reshape(np.array(training_set['actions'][il[0]][start_idx:end_idx]), (-1, 1))
                    sa_pairs = np.concatenate([states, actions], axis=1).astype(np.uint8)

                    sub_pre_states_batch.append(sa_pairs)
                    sub_state_batch.append(training_set['states'][il[0]][end_idx])
                    sub_action_labels.append(training_set['actions'][il[0]][end_idx])
                    sub_reward_labels.append(training_set['rewards'][il[0]])
                    sub_num_states.append(end_idx - start_idx)

                if len(sub_pre_states_batch) > batch_size - 1:
                    max_len_state = 0
                    for shb in sub_pre_states_batch:
                        if shb.shape[0] > max_len_state:
                            max_len_state = shb.shape[0]
                    for idx in range(len(sub_pre_states_batch)):
                        diff_len = max_len_state - sub_pre_states_batch[idx].shape[0]
                        if diff_len != 0:
                            sub_pre_states_batch[idx] = np.concatenate(
                                (sub_pre_states_batch[idx], np.zeros((diff_len, sub_pre_states_batch[idx].shape[1]))), axis=0)

                    pre_states_batches.append(np.array(sub_pre_states_batch, dtype=np.uint8))
                    state_batch.append(np.array(sub_state_batch, dtype=np.uint8))
                    action_labels.append(np.array(sub_action_labels, dtype=np.uint8))
                    reward_labels.append(np.array(sub_reward_labels, dtype=np.int32))
                    num_states.append(np.array(sub_num_states, dtype=np.int32))
                    sub_pre_states_batch = []
                    sub_state_batch = []
                    sub_action_labels = []
                    sub_reward_labels = []
                    sub_num_states = []

                break

    if sub_pre_states_batch != []:
        max_len_state = 0
        for shb in sub_pre_states_batch:
            if shb.shape[0] > max_len_state:
                max_len_state = shb.shape[0]
        for idx in range(len(sub_pre_states_batch)):
            diff_len = max_len_state - sub_pre_states_batch[idx].shape[0]
            if diff_len != 0:
                sub_pre_states_batch[idx] = np.concatenate(
                    (sub_pre_states_batch[idx], np.zeros((diff_len, sub_pre_states_batch[idx].shape[1]))), axis=0)
        pre_states_batches.append(np.array(sub_pre_states_batch, dtype=np.uint8))
        state_batch.append(np.array(sub_state_batch, dtype=np.uint8))
        action_labels.append(np.array(sub_action_labels, dtype=np.uint8))
        reward_labels.append(np.array(sub_reward_labels, dtype=np.int32))
        num_states.append(np.array(sub_num_states, dtype=np.int32))

    return pre_states_batches, state_batch, action_labels, reward_labels, num_states


def gen_eval_batch(test_set, batch_size):
    # obs_size = len(test_set['states'])
    # obs_index = [i for i in range(obs_size)]
    # random.shuffle(obs_index)
    obs_len = len(test_set['states'][0][0])

    pre_states_batches = []
    sub_pre_states_batch = []
    state_batch = []
    sub_state_batch = []
    action_labels = []
    sub_action_labels = []
    reward_labels = []
    sub_reward_labels = []
    num_states = []
    sub_num_states = []

    for idx in range(len(test_set['states'])):
        for idx2 in range(len(test_set['states'][idx])):
            if idx2 == 0:
                sa_pairs = np.zeros((1, obs_len + 1), dtype=np.uint8)
                sub_pre_states_batch.append(sa_pairs)
                sub_state_batch.append(test_set['states'][idx][0])
                sub_action_labels.append(test_set['actions'][idx][0])
                sub_reward_labels.append(test_set['rewards'][idx])
                sub_num_states.append(1)
            else:
                start_idx = 0
                end_idx = idx2
                states = np.array(test_set['states'][idx][start_idx:end_idx])
                actions = np.reshape(np.array(test_set['actions'][idx][start_idx:end_idx]), (-1, 1))
                sa_pairs = np.concatenate([states, actions], axis=1).astype(np.uint8)

                sub_pre_states_batch.append(sa_pairs)
                sub_state_batch.append(test_set['states'][idx][end_idx])
                sub_action_labels.append(test_set['actions'][idx][end_idx])
                sub_reward_labels.append(test_set['rewards'][idx])
                sub_num_states.append(end_idx - start_idx)

            if len(sub_pre_states_batch) > batch_size - 1:
                max_len_state = 0
                for shb in sub_pre_states_batch:
                    if shb.shape[0] > max_len_state:
                        max_len_state = shb.shape[0]
                for idx3 in range(len(sub_pre_states_batch)):
                    diff_len = max_len_state - sub_pre_states_batch[idx3].shape[0]
                    if diff_len != 0:
                        sub_pre_states_batch[idx3] = np.concatenate(
                            (sub_pre_states_batch[idx3], np.zeros((diff_len, sub_pre_states_batch[idx3].shape[1]))), axis=0)
                pre_states_batches.append(np.array(sub_pre_states_batch, dtype=np.uint8))
                state_batch.append(np.array(sub_state_batch, dtype=np.uint8))
                action_labels.append(np.array(sub_action_labels, dtype=np.uint8))
                reward_labels.append(np.array(sub_reward_labels, dtype=np.int32))
                num_states.append(np.array(sub_num_states, dtype=np.int32))
                sub_pre_states_batch = []
                sub_state_batch = []
                sub_action_labels = []
                sub_reward_labels = []
                sub_num_states = []

        if sub_pre_states_batch != []:
            max_len_state = 0
            for shb in sub_pre_states_batch:
                if shb.shape[0] > max_len_state:
                    max_len_state = shb.shape[0]
            for idx in range(len(sub_pre_states_batch)):
                diff_len = max_len_state - sub_pre_states_batch[idx].shape[0]
                if diff_len != 0:
                    sub_pre_states_batch[idx] = np.concatenate(
                        (sub_pre_states_batch[idx], np.zeros((diff_len, sub_pre_states_batch[idx].shape[1]))), axis=0)
            pre_states_batches.append(np.array(sub_pre_states_batch, dtype=np.uint8))
            state_batch.append(np.array(sub_state_batch, dtype=np.uint8))
            action_labels.append(np.array(sub_action_labels, dtype=np.uint8))
            reward_labels.append(np.array(sub_reward_labels, dtype=np.int32))
            num_states.append(np.array(sub_num_states, dtype=np.int32))

    return pre_states_batches, state_batch, action_labels, reward_labels, num_states


def gen_history_sequence(training_set):
    past_traj = []
    num_states = []

    for i in range(len(training_set['states'])):
        states = np.array(training_set['states'][i])
        actions = np.reshape(np.array(training_set['actions'][i]), (-1, 1))
        sa_pairs = np.concatenate([states, actions], axis=1).astype(np.uint8)
        past_traj.append(sa_pairs)
        num_states.append(len(states))

    max_len_state = 0
    for pt in past_traj:
        if pt.shape[0] > max_len_state:
            max_len_state = pt.shape[0]
    for idx in range(len(past_traj)):
        diff_len = max_len_state - past_traj[idx].shape[0]
        if diff_len != 0:
            past_traj[idx] = np.concatenate(
                (past_traj[idx], np.zeros((diff_len, past_traj[idx].shape[1]))), axis=0)

    return np.array(past_traj, dtype=np.uint8), np.array(num_states, dtype=np.int32)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
