import  pickle as pkl 
import numpy as np
import os 
import imageio


# depth_test_path = '/data/ybyang/KITTI-360/data_2d_depth/2013_05_28_drive_0000_sync/image_00/depth_rect/0000000320.png'

# depth_map = imageio.imread(depth_test_path)

# print('done')


root = '/data/ybyang/KITTI-360/semantic_voxel/2013_05_28_drive_0000_sync/(H:16:64,W64:64,L64:64)'

path_list = os.listdir(root)
path_list = [os.path.join(root, n) for n in path_list if n[-4:] == '.npy']
for path in path_list:
    a = np.load(path, allow_pickle=True)
    v = a.tolist()
    path_new = path[:-4] + '.pkl'
    with open(path_new, 'wb+') as fp:
        pkl.dump(v, fp)

print('done')
# seq_list = [0,2,3,4,5,6,7,9,10]
# N = 100000
# os.mkdir(os.path.join('/data/ybyang/KITTI-360', 'trajectory'))
# for seq in seq_list:
#     frames_trajectory_path = os.path.join('/data/ybyang/KITTI-360', 'layout', '2013_05_28_drive_%04d_sync'%(seq) , 'trajectory.pkl')
#     frames_trajectory_path_ = os.path.join('/data/ybyang/KITTI-360', 'trajectory','2013_05_28_drive_%04d_sync.pkl'%(seq))

#     with open(frames_trajectory_path, 'rb') as f: 
#         seq_trajectory = pkl.load(f)
#     seq_trajectory_ = {}
#     for frame in seq_trajectory:
#         seq_trajectory_[frame + seq * N ] = {}
#         for k in seq_trajectory[frame]:
#             seq_trajectory_[frame + seq * N][k + seq * N] = seq_trajectory[frame][k]

#     with open(frames_trajectory_path_, 'wb+') as f: 
#         pkl.dump(seq_trajectory_,f)

#     print('a')

