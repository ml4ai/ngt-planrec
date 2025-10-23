from pathlib import Path
import csv
import numpy as np

RESOURCES_DIR = (Path(__file__).parent / './envs/resources').resolve()
trigger_areas_cvs_path = Path(RESOURCES_DIR, 'MapInfo.csv')
trigger_areas_dict = {}

with open(trigger_areas_cvs_path)as f:
    f_csv = csv.reader(f)
    # print(len(f.readlines()))

    for row in f_csv:
        if row[0] != 'LocationXYZ':
            # trigger_areas_dict.append((int(row[0].split(" ")[0]) + 2111, int(row[0].split(" ")[-1]) - 141))
            trigger_areas_dict[(int(row[0].split(" ")[0]) + 2111, int(row[0].split(" ")[-1]) - 141)] = row[-1]
            # trigger_areas_dict[row[-1]] = (int(row[0].split(" ")[0]) + 2111, int(row[0].split(" ")[-1]) - 141)
            # print(int(row[0].split(" ")[0]) + 2111, int(row[0].split(" ")[-1]) - 141 , row[-1])

    tmp = np.array([24, 11])
    print(trigger_areas_dict[(tmp[0], tmp[1])])
    print((tmp[0], tmp[1]) in trigger_areas_dict.keys())