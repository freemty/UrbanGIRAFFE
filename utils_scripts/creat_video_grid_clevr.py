import os
import cv2
import imageio
import numpy as np 
import matplotlib.pyplot as plt


root_path = '/data/ybyang/out/img/render/clevr/urbangiraffe_clevr'

# 'clevr':[1789,1890,2893,3400,1670,6227,8062,272,1480,1354,1788,7634]

# scene_list = [1789,1890,2893,3400,1670,6227,8062,1480,1354,1788]
# scene_list = [2100,2102,2104,\,\,2107,2108,\,2110,2111,
#               2112,\,2114,2115,2116,\,\,2119,\,2121,2122,2123,2124,2125,2126,2127,2128,\]
scene_list = [2104,2107,2108,
              2110,2111,2114,
              2131,2119,2121,
              2122,2124,2125,
              2126,1789,1890,
              1670,6227,8062,
              1480,1354,1788]
for task_name in ['clevr_video']:
    # task_name =  'object_editing_video'
    video_name = os.path.join(root_path,  task_name + '_final.avi')
    fps = 5
    h_num, w_num = 3, 7
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