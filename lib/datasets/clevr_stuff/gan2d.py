import numpy as np
import os
import random
from lib.config import cfg, args
import imageio
import cv2


name2id = {'ground' : 1,
              'wall' : 2,
              'object' : 26 }
name2color = {'ground' : 1,
              'wall' : 1,
              'object' : 2 }
class Dataset:
    def __init__(self, data_root, seq_list, split):
        super(Dataset, self).__init__()
        # path and initialization
        self.data_root = data_root
        self.split = split
        self.use_depth = cfg.use_depth

        self.ratio = cfg.ratio
        self.height, self.width = 256, 256
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)
        idx2frame = []

        self.frame_id_list = []
        self.rgb_image_dict = {}

        N = 1000
        for seq_num in seq_list:
            metadata_dir = os.path.join(data_root, '%d'%seq_num, 'Metadata')
            self.metadata_dict = {int(s[16:21]) : os.path.join(metadata_dir, s) for s in os.listdir(metadata_dir)}
            for idx in self.metadata_dict:
                rgb_path = os.path.join(data_root, '%d'%seq_num, 'RGB', 'CLEVRTEX_train_%06d.png'%(idx))
                if (not os.path.exists(rgb_path)):
                    continue
                self.rgb_image_dict[idx] = rgb_path
                self.frame_id_list += [idx]
                    

            print('Load clevr Stuff seq%d Done!!'%seq_num)

    
        frame_id_list_gt = self.frame_id_list.copy()
        frame_id_list_fake = self.frame_id_list.copy()

        random.shuffle(frame_id_list_gt)
        random.shuffle(frame_id_list_fake)

        self.frame_id_list_gt =frame_id_list_gt
        self.frame_id_list_fake = frame_id_list_fake

        self.idx2frame = {idx : frame  for idx,frame in enumerate(self.frame_id_list)}
        self.frame2idx = {frame : idx  for idx,frame in enumerate(self.frame_id_list)}


        print('Load dataset done!')

    @staticmethod
    def load_img(path, W = 352, H = 94, type = 'rgb', sr_multiple = 1):
        image = imageio.imread(path).astype(np.float32)
        if type == 'rgb':
            mean=127.5
            std=127.5
            rgb_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)[...,:3]
            rgb_image = rgb_image/ std  - 1
            return rgb_image
        elif type == 'depth':
            z_near, z_far = 0.001, cfg.z_far
            depth_image = cv2.resize(image, (W // sr_multiple, H // sr_multiple), interpolation=cv2.INTER_AREA).astype(np.float32)
            depth_image = cv2.resize(depth_image, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            # print(np.max(depth_image))
            depth_image = (depth_image - z_near) / (z_far - z_near)
            depth_image = np.clip(depth_image, 0., 1.)
            return np.expand_dims(depth_image, -1)
        else:
            raise KeyError

    def __getitem__(self, index):

        # Load scene for G training
        frame_id = self.frame_id_list[index]
        # idx_fake = self.frame2idx[frame_id_fake]

        # Load gt image for D training
        frame_id= self.frame_id_list[index]
        rgb = self.rgb_image_dict[frame_id]

        ret = {
            'frame_id': frame_id ,
            'frame_id_gt': frame_id ,
            'rgb': self.load_img(rgb ,W = self.W, H = self.H, type ='rgb').transpose(2,0,1),
        }
        ret['rgb_gt'] = ret['rgb']
        return ret

    def __len__(self):
        return len(self.frame_id_list)
