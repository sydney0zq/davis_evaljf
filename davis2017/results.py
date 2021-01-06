import os
import numpy as np
from PIL import Image
import sys
from multiprocessing import Pool



    
def read_mask_plain(x):
    root_dir, sequence, frame_id = x
    mask_path = os.path.join(root_dir, sequence, f'{frame_id}.png')
    return np.array(Image.open(mask_path))

class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pool = Pool(8)

    def _read_mask(self, sequence, frame_id):
        try:
            mask_path = os.path.join(self.root_dir, sequence, f'{frame_id}.png')
            return np.array(Image.open(mask_path))
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        num_objects = int(np.max(self._read_mask(sequence, "00000")))
        mask_0 = self._read_mask(sequence, masks_id[0])
        # single process
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)

        # multiple process
        # masks_list = self.pool.map(read_mask_plain, [[self.root_dir, sequence, m] for m in masks_id])
        # masks = np.concatenate([x[None, ...] for x in masks_list])


        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks


if __name__ == "__main__":
    r = Results(root_dir = "../../experiments/davis17_visual")
    r.read_masks("bike-packing", ["00000"])
