import os 
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from tqdm import tqdm
from .fid import calculate_activation_statistics, calculate_frechet_distance
from .fid import calculate_kid_given_paths

'''
NOTE: The code is largely adopted from:
https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
'''

class Evaluator:
    def __init__(self,):
        self.frames = []
        self.eval_dict = {}
        self.label_dict = []
        self.gt_list = []
        self.pred_list = []
    
    def evaluate_fid(self, gt_dir, pred_dir, tag = 'obj'):
        fake_files = [os.path.join(pred_dir, f) for f in  os.listdir(pred_dir)]
        gt_files = [os.path.join(gt_dir, f) for f in  os.listdir(gt_dir)]
        
        # Calc KID
        kid_score = calculate_kid_given_paths((gt_dir,pred_dir))[0][1]
        # Calc FID
        mu, sigma = calculate_activation_statistics(fake_files)
        mu_gt, sigma_gt = calculate_activation_statistics(gt_files)
        fid_score = calculate_frechet_distance(
            mu, sigma, mu_gt, sigma_gt, eps=1e-4)


        eval_dict = {
            'fid_' + tag: fid_score,
            'kid_' + tag: kid_score
        }

        return eval_dict


        # fid_score = calculate_frechet_distance(img_dir)
        activation_statistics = calculate_activation_statistics(img_dir)


    # def summarize(self):
    #     miou_list = []
    #     gt_all = np.concatenate(self.gt_list)
    #     pred_all = np.concatenate(self.pred_list)
    #     label_all_dict = defaultdict(list)
    #     mIoUs = compute_mIoU(pred_all, gt_all, len(label))
    #     for i in range(len(mIoUs)):
    #         if mIoUs[i] > 0.1 and mIoUs[i] < 1:
    #             label_all_dict[label2name[label[i]]].append(mIoUs[i])
    #     mask = (mIoUs>0.1) & (mIoUs<1)
    #     mIoUs = mIoUs[mask]
    #     for item in label_all_dict.values():
    #         miou_list.append(item)
    #     print('miou: {}'.format(np.mean(np.array(mIoUs))))
    #     for i in label_all_dict.keys():
    #         print('IoU of {0} is {1}'.format(i, np.mean(label_all_dict[i])))
    #     print('total acc:{}'.format((pred_all==gt_all).sum()/len(gt_all)))


