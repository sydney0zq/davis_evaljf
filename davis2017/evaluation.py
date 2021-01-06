import sys
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from collections import OrderedDict
from .davis import DAVIS
from .metrics import db_eval_boundary, db_eval_iou
from . import utils
from .results import Results

from scipy.optimize import linear_sum_assignment
from concurrent import futures


class DAVISEvaluation(object):
    def __init__(self, davis_root, task, gt_set, sequences='all', codalab=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, task=task, subset=gt_set, sequences=sequences, use_pickle=False, codalab=codalab)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        # sys.stdout.write("=> start to evaluate semi supervised results...")
        if all_res_masks.shape[0] != all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        # sys.stdout.write ("=> the values of J and F metrics are computed...")
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_semisupervised_parallel(all_gt_masks, all_res_masks, all_void_masks, metric):
        # sys.stdout.write("=> start to evaluate semi supervised results...")
        if all_res_masks.shape[0] != all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        num_threads = 4
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])

        J_fs = []
        F_fs = []
        # 进程fork开销很大，8 seqs: 84s -> 101s，不建议使用
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            for ii in range(all_gt_masks.shape[0]):
                J_fs.append(executor.submit(db_eval_iou, all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks))
                F_fs.append(executor.submit(db_eval_boundary, all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks))

        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = J_fs[ii]._result
            if 'F' in metric:
                f_metrics_res[ii, :] = F_fs[ii]._result
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    def evaluate(self, res_path, metric=('J', 'F')):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        seq_info = OrderedDict()
        for seq in list(self.dataset.get_sequences()):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            if self.task == 'unsupervised':
                j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
            elif self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)

            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM
                seq_info.update({seq: {'num_objects': all_gt_masks.shape[0]}})
            # Show progress
        return metrics_res


    def evaluate_parallel(self, res_path, metric=('J', 'F')):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)

        num_threads = 16
        JF_res = []
        seq_info = OrderedDict()

        # ProcessPoolExecutor
        # 这个模块实现的是真正的并行计算，因为它使用ProcessPoolExecutor类把工作分配给多个Python进程处理。
        # 因此，如果需要做CPU 密集型处理，使用这个模块能绕开GIL，利用所有可用的CPU核心。
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            for seq in list(self.dataset.get_sequences())[:8]:
                all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
                if self.task == 'semi-supervised':
                    all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
                all_res_masks = results.read_masks(seq, all_masks_id)

                if self.task == 'unsupervised':
                    JF_res.append(executor.submit(self._evaluate_unsupervised, all_gt_masks, all_res_masks, all_void_masks, metric))
                elif self.task == 'semi-supervised':
                    # func_args.append([])
                    JF_res.append(executor.submit(self._evaluate_semisupervised, all_gt_masks, all_res_masks, None, metric))
                print({seq_obj_name: {'JMean': JM, 'FMean': FM}})

        for seq_id, (seq, seq_info) in enumerate(seq_info.items()):
            num_objects = seq_info['num_objects']
            seq_JF_res = JF_res[seq_id]._result

            for ii in range(num_objects):
                seq_obj_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(seq_JF_res[0][ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_obj_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(seq_JF_res[1][ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_obj_name] = FM
                print({seq_obj_name: {'JMean': JM, 'FMean': FM}})
        return metrics_res
