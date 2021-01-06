#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

import time
import pandas as pd
import sys, os
import numpy as np

CURR_DIR = os.path.dirname(__file__)
from davis2017.evaluation import DAVISEvaluation

DAVIS_ROOT = os.path.join(CURR_DIR, "DAVIS2017-trainval")
RES_PATH = sys.argv[1]
IMSET = "DAVIS17val"
csv_name_global_path = f'{RES_PATH}/global_results-{IMSET}.csv'
csv_name_per_sequence_path = f'{RES_PATH}/per-sequence_results-{IMSET}.csv'

time_start = time.time()
dataset_eval = DAVISEvaluation(davis_root=DAVIS_ROOT, task="semi-supervised", gt_set="val")
metrics_res = dataset_eval.evaluate(RES_PATH)
# metrics_res = dataset_eval.evaluate_parallel(RES_PATH)
J, F = metrics_res['J'], metrics_res['F']

# Generate dataframe for the general results
g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                          np.mean(F["D"])])
g_res = np.reshape(g_res, [1, len(g_res)])
table_g = pd.DataFrame(data=g_res, columns=g_measures)
with open(csv_name_global_path, 'w') as f:
    table_g.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_name_global_path}')

# Generate a dataframe for the per sequence results
seq_names = list(J['M_per_object'].keys())
seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
J_per_object = [J['M_per_object'][x] for x in seq_names]
F_per_object = [F['M_per_object'][x] for x in seq_names]
table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
with open(csv_name_per_sequence_path, 'w') as f:
    table_seq.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {IMSET} ---------------------------\n")
print(table_g.to_string(index=False))
#sys.stdout.write(f"\n---------- Per sequence results for {IMSET} ----------\n")
#print(table_seq.to_string(index=False))
total_time = time.time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))


