import os
import cv2
import imageio
import numpy as np 
import matplotlib.pyplot as plt

def make_video(img_list_dir, name = None, save_dir = None, fps = 10):
    
    video_name = os.path.join(save_dir, name + '.avi')
    images= [os.path.join(img_list_dir, f) for f in os.listdir(img_list_dir)]
    images.sort()
    # images = [img for img in if img.endswith(".png")]
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


root_path = '/data/ybyang/out/img/render/kitti-360/urbangiraffe_kitti360'

# ['interpolate_camera_video'8, 'move_forward_video'7,'building_lower_video','road2grass_video','building2tree_video','object_editing_video'10]

# Done 
# 'road2grass_video':3400,1670,6227,8062,272,354,6950,5400,7334,7634,911358,502364,604197,210235,5287
# 'interpolate_camera_video':[272,1670,3400,6950,7334,7634,8062,207712,210235,405074,604197, 6636,1733,3368,207717]
# 'move_forward_video':[6636,1733,3368,205327,206499,1670,6227,8062,272,6950,788,7334,7634,502364,210235,207712]
# 'building2tree_video': [1733,6636,3368,3400,1670,6227,8062,354,6950,5400,5287,10697,3386,6783,7895]
# 'building_lower_video': [1733,3400,1670,6227,6950,8062,354,5400,7634,911358,604197,405074,8886,8014,205151]
# 'object_editing':[6636,1733,3368,205327,206499,1670,6227,8062,272,6950,788,7334,7634,502364,3400,207712]

# 'clevr':[1789,1890,2893,3400,1670,6227,8062,1480,1354,1788,7634]

scene_list = [1733,3368,205327,206499,1670,6227,8062,272,6950,502364,7334,7634,502364,210235,207712]
for task_name in ['move_forward_video']:
    # task_name =  'object_editing_video'
    video_name = os.path.join(root_path,  task_name + '_final.avi')
    fps = 10
    h_num, w_num = 5, 3
    assert len(scene_list) == h_num * w_num

    images_list = []
    for i, scene in enumerate(scene_list):
        scene_dir = os.path.join(root_path, task_name, '%06d'%scene, 'rgb')
        scene_frame_list = os.listdir(scene_dir)
        scene_frame_list.sort()
        images_list += [[os.path.join(scene_dir, img) for img in scene_frame_list]]

    images_list = np.array(images_list)
    frame_num = images_list.shape[1]
    img = cv2.imread(images_list[0][0])
    height, width = img.shape[0],img.shape[1]
    video = cv2.VideoWriter(video_name, 0, fps, (width*w_num,height*h_num))

    for f in range(frame_num):
        big_frame = np.zeros((height * h_num, width * w_num, 3), dtype=np.uint8)
        for i in range(w_num):
            for j in range(h_num):
                big_frame[j * height:(j+1) * height, i * width:(i+1) * width] = cv2.imread(images_list[i * h_num + j, f])
        video.write(big_frame)

    cv2.destroyAllWindows()
    video.release()