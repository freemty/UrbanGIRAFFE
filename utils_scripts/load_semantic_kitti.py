import os 

import numpy as np
import pickle as pkl


data_root = '/data/ybyang/semantic-kitti/'
with open(os.path.join(data_root, '000010.pkl'), 'rb') as f:
    a = pkl.load(f)
print('Done')

