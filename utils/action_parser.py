'''
parse moving and triage actions from human subjects data
'''

import json
import numpy as np
import math
from dateutil.parser import parse
from pathlib import Path
import copy

RESOURCES_DIR = (Path(__file__).parent / '../data').resolve()
invalid_data_found = False

# file_name = 'HSRData_TrialMessages_CondBtwn-NoTriageNoSignal_CondWin-FalconEasy-StaticMap_Trial-120_Team-na_Member-51_Vers-1.metadata'
# file_path = Path(RESOURCES_DIR, file_name)

actions = ['move left (0)',
           'move right (1)',
           'move up (2)',
           'move down (3)',
           'triage victim (4)',
           'no action (5)'
           ]
time_costs = [0.15, 0.15, 0.15, 0.15, 7.5, 1]
TOTAL_TIME = 10 * 60


# process data into integers and ignore the redundant states
def process_data(data):
    processed_data = []
    for d in data:
        m, s = d['mission_timer'].strip().split(":")
        timer = TOTAL_TIME - (int(m) * 60 + int(s))
        if processed_data:
            if timer < processed_data[-1]['mission_timer']:
                continue
        if d['action'] == 'move':
            # m, s = d['mission_timer'].strip().split(":")
            # timer = TOTAL_TIME - (int(m) * 60 + int(s))
            loc_x = math.floor(d['x'])
            loc_z = math.floor(d['z'])

            if processed_data != [] and processed_data[-1]['action'] == 'move':
                if loc_x == processed_data[-1]['x'] and loc_z == processed_data[-1]['z'] or loc_x <= -2150:
                    continue
                else:
                    d_tmp = {'action': 'move', 'mission_timer': timer, 'x': loc_x, 'z': loc_z}
                    processed_data.append(d_tmp)
            elif processed_data != [] and processed_data[-1]['action'] == 'triage':
                if processed_data[-1]['triage_state'] != 'IN_PROGRESS' and loc_x > -2150:
                    d_tmp = {'action': 'move', 'mission_timer': timer, 'x': loc_x, 'z': loc_z}
                    processed_data.append(d_tmp)
                # 'successful' and 'unsuccessful' state may miss in human subjects data
                elif processed_data[-1]['triage_state'] == 'IN_PROGRESS' and (
                        abs(loc_x - processed_data[-1]['victim_x']) + abs(loc_z - processed_data[-1]['victim_z'])) > 6:
                    d_tmp = copy.deepcopy(processed_data[-1])
                    d_tmp['mission_timer'] = timer
                    d_tmp['triage_state'] = 'UNSUCCESSFUL'
                    processed_data.append(d_tmp)
                elif processed_data[-1]['triage_state'] == 'IN_PROGRESS' and processed_data[-1]['color'] == 'Yellow':
                    if timer - processed_data[-1]['mission_timer'] > 16:
                        d_tmp = copy.deepcopy(processed_data[-1])
                        d_tmp['mission_timer'] = timer
                        d_tmp['triage_state'] = 'SUCCESSFUL'
                        processed_data.append(d_tmp)
                elif processed_data[-1]['triage_state'] == 'IN_PROGRESS' and processed_data[-1]['color'] == 'Green':
                    if timer - processed_data[-1]['mission_timer'] > 8:
                        d_tmp = copy.deepcopy(processed_data[-1])
                        d_tmp['mission_timer'] = timer
                        d_tmp['triage_state'] = 'SUCCESSFUL'
                        processed_data.append(d_tmp)

            elif processed_data == [] and loc_x > -2150:
                d_tmp = {'action': 'move', 'mission_timer': timer, 'x': loc_x, 'z': loc_z}
                processed_data.append(d_tmp)
            elif loc_x < -2150:
                continue
        else:
            # m, s = d['mission_timer'].strip().split(":")
            # timer = TOTAL_TIME - (int(m) * 60 + int(s))
            loc_x = math.floor(d['victim_x'])
            loc_z = math.floor(d['victim_z'])
            d_tmp = {'action': 'triage', 'mission_timer': timer, 'victim_x': loc_x, 'victim_z': loc_z,
                     'color': d['color'], 'triage_state': d['triage_state']}
            processed_data.append(d_tmp)

    return processed_data


# data to action
def data_to_action(data, file_name):
    # load map according to the difficulty
    if 'FalconEasy' in file_name:
        map_path = Path(RESOURCES_DIR, 'Falcon_easy.npy')
    elif 'FalconMed' in file_name:
        map_path = Path(RESOURCES_DIR, 'Falcon_medium.npy')
    elif 'FalconHard' in file_name:
        map_path = Path(RESOURCES_DIR, 'Falcon_difficult.npy')
    else:
        map_path = Path(RESOURCES_DIR, 'raw_map_state.npy')

    gt_map = np.load(map_path)
    # initial position and direction
    pre_loc_x = pre_loc_z = 5
    pre_timer = 0

    current_timer = 0
    action_sequence = []
    global invalid_data_found

    saved_green_num = saved_yellow_num = 0
    episode_reward = 0

    for d in data:
        timer = d['mission_timer']
        if timer - pre_timer > 20:
            # print('missing data found!')
            invalid_data_found = True
            break
        else:
            pre_timer = timer

        if d['action'] == 'move':
            loc_x = d['x'] + 2110
            loc_z = d['z'] - 142

            if abs(pre_loc_x - loc_x) + abs(pre_loc_z - loc_z) > 20:
                # print('missing data found!')
                invalid_data_found = True
                break

            if gt_map[loc_z, loc_x] == 4:
                continue

            # move to the current location
            current_loc_x, current_loc_z = pre_loc_x, pre_loc_z
            diff_x = loc_x - pre_loc_x
            diff_z = loc_z - pre_loc_z
            if diff_x != 0:
                if diff_x > 0:
                    while diff_x != 0:
                        current_loc_x += 1
                        if gt_map[current_loc_z, current_loc_x] == 4:
                            current_loc_x -= 1
                            if diff_z != 0:
                                pre_diff_z = diff_z
                                if diff_z > 0:
                                    while diff_z != 0:
                                        current_loc_z += 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_z -= 1
                                            break
                                        action = 3
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_z -= 1
                                else:
                                    while diff_z != 0:
                                        current_loc_z -= 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_z += 1
                                            break
                                        action = 2
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_z += 1
                                if pre_diff_z != diff_z:
                                    continue
                                else:
                                    break
                            else:
                                break
                        action = 1
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                        diff_x -= 1
                else:
                    while diff_x != 0:
                        current_loc_x -= 1
                        if gt_map[current_loc_z, current_loc_x] == 4:
                            current_loc_x += 1
                            if diff_z != 0:
                                pre_diff_z = diff_z
                                if diff_z > 0:
                                    while diff_z != 0:
                                        current_loc_z += 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_z -= 1
                                            break
                                        action = 3
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_z -= 1
                                else:
                                    while diff_z != 0:
                                        current_loc_z -= 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_z += 1
                                            break
                                        action = 2
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_z += 1
                                if pre_diff_z != diff_z:
                                    continue
                                else:
                                    break
                            else:
                                break
                        action = 0
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                        diff_x += 1

            if diff_z != 0:
                if diff_z > 0:
                    while diff_z != 0:
                        current_loc_z += 1
                        if gt_map[current_loc_z, current_loc_x] == 4:
                            current_loc_z -= 1
                            if diff_x != 0:
                                pre_diff_x = diff_x
                                if diff_x > 0:
                                    while diff_x != 0:
                                        current_loc_x += 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_x -= 1
                                            break
                                        action = 1
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_x -= 1
                                else:
                                    while diff_x != 0:
                                        current_loc_x -= 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_x += 1
                                            break
                                        action = 0
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_x += 1
                                if pre_diff_x != diff_x:
                                    continue
                                else:
                                    break
                            else:
                                break
                        action = 3
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                        diff_z -= 1
                else:
                    while diff_z != 0:
                        current_loc_z -= 1
                        if gt_map[current_loc_z, current_loc_x] == 4:
                            current_loc_z += 1
                            if diff_x != 0:
                                pre_diff_x = diff_x
                                if diff_x > 0:
                                    while diff_x != 0:
                                        current_loc_x += 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_x -= 1
                                            break
                                        action = 2
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_x -= 1
                                else:
                                    while diff_x != 0:
                                        current_loc_x -= 1
                                        if gt_map[current_loc_z, current_loc_x] == 4:
                                            current_loc_x += 1
                                            break
                                        action = 0
                                        action_sequence.append(action)
                                        current_timer += time_costs[action]
                                        diff_x += 1
                                if pre_diff_x != diff_x:
                                    continue
                                else:
                                    break
                            else:
                                break
                        action = 2
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                        diff_z += 1

            if diff_x != 0 or diff_z != 0:
                # print('invalid move!')
                invalid_data_found = True
                break

            pre_loc_x, pre_loc_z = loc_x, loc_z

        else:
            if d['triage_state'] == 'IN_PROGRESS':
                # the agent should move to the closest location to the victim and face to it
                victim_x = d['victim_x'] + 2110
                victim_z = d['victim_z'] - 142

                # need to move first
                if pre_loc_x != victim_x or pre_loc_z != victim_z:
                    loc_x, loc_z = victim_x, victim_z

                    # move to the current location
                    current_loc_x, current_loc_z = pre_loc_x, pre_loc_z
                    diff_x = loc_x - pre_loc_x
                    diff_z = loc_z - pre_loc_z
                    if diff_x != 0:
                        if diff_x > 0:
                            while diff_x != 0:
                                current_loc_x += 1
                                if gt_map[current_loc_z, current_loc_x] == 4:
                                    current_loc_x -= 1
                                    if diff_z != 0:
                                        pre_diff_z = diff_z
                                        if diff_z > 0:
                                            while diff_z != 0:
                                                current_loc_z += 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_z -= 1
                                                    break
                                                action = 3
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_z -= 1
                                        else:
                                            while diff_z != 0:
                                                current_loc_z -= 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_z += 1
                                                    break
                                                action = 2
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_z += 1
                                        if pre_diff_z != diff_z:
                                            continue
                                        else:
                                            break
                                    else:
                                        break
                                action = 1
                                action_sequence.append(action)
                                current_timer += time_costs[action]
                                diff_x -= 1
                        else:
                            while diff_x != 0:
                                current_loc_x -= 1
                                if gt_map[current_loc_z, current_loc_x] == 4:
                                    current_loc_x += 1
                                    if diff_z != 0:
                                        pre_diff_z = diff_z
                                        if diff_z > 0:
                                            while diff_z != 0:
                                                current_loc_z += 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_z -= 1
                                                    break
                                                action = 3
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_z -= 1
                                        else:
                                            while diff_z != 0:
                                                current_loc_z -= 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_z += 1
                                                    break
                                                action = 2
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_z += 1
                                        if pre_diff_z != diff_z:
                                            continue
                                        else:
                                            break
                                    else:
                                        break
                                action = 0
                                action_sequence.append(action)
                                current_timer += time_costs[action]
                                diff_x += 1

                    if diff_z != 0:
                        if diff_z > 0:
                            while diff_z != 0:
                                current_loc_z += 1
                                if gt_map[current_loc_z, current_loc_x] == 4:
                                    current_loc_z -= 1
                                    if diff_x != 0:
                                        pre_diff_x = diff_x
                                        if diff_x > 0:
                                            while diff_x != 0:
                                                current_loc_x += 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_x -= 1
                                                    break
                                                action = 1
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_x -= 1
                                        else:
                                            while diff_x != 0:
                                                current_loc_x -= 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_x += 1
                                                    break
                                                action = 0
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_x += 1
                                        if pre_diff_x != diff_x:
                                            continue
                                        else:
                                            break
                                    else:
                                        break
                                action = 3
                                action_sequence.append(action)
                                current_timer += time_costs[action]
                                diff_z -= 1
                        else:
                            while diff_z != 0:
                                current_loc_z -= 1
                                if gt_map[current_loc_z, current_loc_x] == 4:
                                    current_loc_z += 1
                                    if diff_x != 0:
                                        pre_diff_x = diff_x
                                        if diff_x > 0:
                                            while diff_x != 0:
                                                current_loc_x += 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_x -= 1
                                                    break
                                                action = 2
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_x -= 1
                                        else:
                                            while diff_x != 0:
                                                current_loc_x -= 1
                                                if gt_map[current_loc_z, current_loc_x] == 4:
                                                    current_loc_x += 1
                                                    break
                                                action = 0
                                                action_sequence.append(action)
                                                current_timer += time_costs[action]
                                                diff_x += 1
                                        if pre_diff_x != diff_x:
                                            continue
                                        else:
                                            break
                                    else:
                                        break
                                action = 2
                                action_sequence.append(action)
                                current_timer += time_costs[action]
                                diff_z += 1

                    if diff_x != 0 or diff_z != 0:
                        # print('invalid move!')
                        invalid_data_found = True
                        break

                    pre_loc_x, pre_loc_z = loc_x, loc_z

            elif d['triage_state'] == 'SUCCESSFUL':
                action = 4
                if d['color'] == 'Green':
                    for _ in range(1):
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                    episode_reward += 10
                    if timer < TOTAL_TIME / 2:
                        saved_green_num += 1
                else:
                    for _ in range(2):
                        action_sequence.append(action)
                        current_timer += time_costs[action]
                    episode_reward += 30
                    if timer < TOTAL_TIME / 2:
                        saved_yellow_num += 1
            else:
                action = 5
                triage_time = int(math.floor((d['mission_timer'] - current_timer) / time_costs[action]))
                triage_time = max(1, triage_time)
                for _ in range(int(triage_time // time_costs[action])):
                    action_sequence.append(action)
                    current_timer += time_costs[action]

                # since this version doesn't contain yaw, we need to add one no action behind the failure triage
                # action = 5
                # action_sequence.append(action)
                # current_timer += time_costs[action]

        if current_timer < timer:
            action = 5
            for _ in range(int((timer - current_timer) / time_costs[action])):
                action_sequence.append(action)
                current_timer += time_costs[action]
    # print(current_timer)
    if data == []:
        invalid_data_found = True

    if current_timer < TOTAL_TIME and data != []:
        if data[-1]['action'] == 'triage' and data[-1]['triage_state'] == 'IN_PROGRESS':
            action = 4
            for _ in range(int(math.ceil(TOTAL_TIME - current_timer) / time_costs[action]) + 1):
                action_sequence.append(action)
                current_timer += time_costs[action]
        else:
            action = 5
            for _ in range(int((TOTAL_TIME - current_timer) / time_costs[action] + 1)):
                action_sequence.append(action)
                current_timer += time_costs[action]

    #  classify the strategy according to the ratio between the numbers of saved green and yellow victims within 5 minutes
    # if saved_yellow_num == 0 or saved_green_num / (saved_green_num + saved_yellow_num) >= 0.75:
    #     strategy = 'green'
    if saved_green_num == 0 or saved_yellow_num / (saved_green_num + saved_yellow_num) >= 0.75:
        strategy = 'yellow'
    else:
        strategy = 'opportunistic'
    return np.array(action_sequence), invalid_data_found, strategy, episode_reward


def json_to_action(file_name):
    file_path = Path(RESOURCES_DIR, file_name)
    # Open the input file, parse each line in it as a JSON-serialized object.
    with open(file_path, "r") as f:
        messages = []
        for line in f:
            jline = None
            try:
                jline = json.loads(line)
            except:
                print(f"Bad json line of len: {len(line)}, {line}")
            if jline is not None and 'topic' in jline.keys():
                if jline['topic'] in ['observations/state', 'observations/events/player/triage']:
                    messages.append(jline)

        sorted_messages = sorted(
            messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        simple_messages = []
        for data in sorted_messages:
            # ignore data before game
            if data['data']['mission_timer'] not in ['Mission Timer not initialized.', '10 : 3', '10 : 2', '10 : 1']:
                if data['topic'] == 'observations/state':
                    data_tmp = {'action': 'move', 'mission_timer': data['data']['mission_timer'],
                                'x': data['data']['x'],
                                'z': data['data']['z'], 'yaw': data['data']['yaw']}
                    simple_messages.append(data_tmp)
                else:
                    data_tmp = {'action': 'triage', 'mission_timer': data['data']['mission_timer'],
                                'victim_x': data['data']['victim_x'], 'victim_z': data['data']['victim_z'],
                                'color': data['data']['color'],
                                'triage_state': data['data']['triage_state']}
                    simple_messages.append(data_tmp)

        processed_data = process_data(simple_messages)
        return data_to_action(processed_data, file_name)
