import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
import pickle
from multiprocessing import Pool

import os, sys
CURR_DIR = os.path.dirname(__file__)



def read_element(x):
    return np.array(Image.open(x))


class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, root, task='unsupervised', subset='val', sequences='all', resolution='480p', use_pickle=False, codalab=False):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        :param use_pickle: Pickle the Annotations of DAVIS as a single pkl file, however, it will costs at least 6G space, not recommend
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root, annotations_folder, resolution)
        year = '2019' if task == 'unsupervised' and (subset == 'test-dev' or subset == 'test-challenge') else '2017'
        self.imagesets_path = os.path.join(self.root, 'ImageSets', year)
        self.year = year

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

        self.pool = Pool(8)

        self.use_pickle = use_pickle
        if use_pickle:
            self.seq_masks = self._cache_dataset()

    def _cache_dataset(self):
        cache_dir = os.path.join(CURR_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        pkl_name = f"davis{self.year}_{self.task}_{self.subset}_mask.pkl"
        cache_fn = os.path.join(cache_dir, pkl_name)
        if os.path.exists(cache_fn):
            print ("DAVIS masks have been cached to {}...".format(cache_fn))
            with open(cache_fn, "rb") as f:
                seq_masks = pickle.load(f)
        else:
            print ("DAVIS masks will be cached to {}...".format(cache_fn))
            seq_masks = {}
            for seq in self.sequences:
                all_objs, obj_id = self._get_all_elements(seq, 'masks')
                all_objs = np.uint8(all_objs)
                seq_masks[seq] = [all_objs, obj_id]
            with open(cache_fn, "wb") as f:
                pickle.dump(seq_masks, f)
        return seq_masks
    
    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'DAVIS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found, download the missing subset '
                                    f'for the {self.task} task from {self.DATASET_WEB}')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder for the {self.task} task not found, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id
    
    def _get_all_elements_parallel(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))

        all_objs_list = self.pool.map(read_element, self.sequences[sequence][obj_type])
        all_objs = np.concatenate([x[None, ...] for x in all_objs_list])
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False, parallel=True):
        if self.use_pickle:
            masks, masks_id = self.seq_masks[sequence]
        else:
            if parallel:
                masks, masks_id = self._get_all_elements_parallel(sequence, 'masks')
            else:
                masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    only_first_frame = True
    subsets = ['train', 'val']

    for s in subsets:
        dataset = DAVIS(root='/home/csergi/scratch2/Databases/DAVIS2017_private', subset=s)
        for seq in dataset.get_sequences():
            g = dataset.get_frames(seq)
            img, mask = next(g)
            plt.subplot(2, 1, 1)
            plt.title(seq)
            plt.imshow(img)
            plt.subplot(2, 1, 2)
            plt.imshow(mask)
            plt.show(block=True)

