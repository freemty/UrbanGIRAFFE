import os
import cv2
import imageio
import numpy as np 
import matplotlib.pyplot as plt



root_path = '/data/ybyang/out/img/render/kitti-360/urbangiraffe_kitti360'


# baseline1_root = '/data/ybyang/out/img/render/kitti-360/gsn_scene'
# baseline2_root = '/data/ybyang/out/img/render/kitti-360/giraffe_kitti360/move_forward_video_compare'
our_root = '/data/ybyang/out/img/render/kitti-360/urbangiraffe_kitti360/move_forward_video_compare'
baseline1_root = '/data/ybyang/out/img/render/kitti-360/urbangiraffe_nopatchD/move_forward_video_compare'
baseline2_root = '/data/ybyang/out/img/render/kitti-360/urbangiraffe_norecon/move_forward_video_compare'

scene_list = [604197,3400,5400,6950,7334]
for task_name in ['move_forward_video_compare']:
    # task_name =  'object_editing_video'
    video_name = os.path.join(our_root,  task_name + '_final_ablation.avi')
    fps = 5
    h_num, w_num = 5, 3
    assert len(scene_list)  == h_num 

    images_list = []
    for root_path in [baseline1_root,baseline2_root ,our_root]:
        for i, scene in enumerate(scene_list):
            scene_dir = os.path.join(root_path,'%06d'%scene, 'rgb')
            scene_frame_list = os.listdir(scene_dir)
            scene_frame_list.sort()
            images_list += [[os.path.join(scene_dir, img) for img in scene_frame_list[:41]]]

    images_list = np.array(images_list)
    frame_num = 39
    img = cv2.imread(images_list[0][0])
    height, width = img.shape[0],img.shape[1]
    video = cv2.VideoWriter(video_name, 0, fps, (width*w_num,height*h_num))

    for f in range(frame_num):
        big_frame = np.zeros((height * h_num, width * w_num, 3), dtype=np.uint8)
        for i in range(w_num):
            for j in range(h_num):
                image = cv2.imread(images_list[i * h_num + j][f])
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                big_frame[j * height:(j+1) * height, i * width:(i+1) * width] = image
        video.write(big_frame)

    cv2.destroyAllWindows()
    video.release()