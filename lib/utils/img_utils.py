
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import os
import imageio
from PIL import Image
from lib.utils import data_config
import torchvision
from torchvision import transforms

def save_tensor_img(img_tnesor, save_dir, name ,type = 'rgb', valid_frame_num = 8):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    toPIL = transforms.ToPILImage()
    img_tnesor = img_tnesor.detach().cpu().detach()

    if len(img_tnesor.shape) == 4:
        img_tnesor =  torchvision.utils.make_grid(img_tnesor)
    if len(img_tnesor.shape) == 5:
        valid_frame_idx = [i for i in range(0, img_tnesor.shape[0], img_tnesor.shape[0] // valid_frame_num)]
        img_tnesor = img_tnesor[valid_frame_idx]
        frame_num, batch_size, C, H, W = img_tnesor.shape
        img_tnesor = img_tnesor.reshape(frame_num * batch_size, C, H , W)
        img_tnesor =  torchvision.utils.make_grid(img_tnesor, nrow=batch_size)

    pic = img_tnesor.numpy()
    if type == 'rgb':
        pic = pic * 127.5 + 127.5
        pic = np.clip(pic,0,255).astype(np.uint8).transpose(1,2,0)
        imageio.imsave(os.path.join(save_dir, name), pic)
    elif type == 'depth':
        cmap = cv2.COLORMAP_JET
        # z_near, z_far = 0, 100
        pic = pic * 255
        pic = np.clip(pic,0,255).astype(np.uint8).transpose(1,2,0)
        pic = cv2.applyColorMap(pic, cmap)
        imageio.imsave(os.path.join(save_dir, name), pic)
    # pic = Image.fromarray(pic, mode='RGB')
    # pic.save(os.path.join(save_dir, name))
        
        

def make_gif(img_list_dir, name = None, save_dir = None, fps = 10):
    
    if name == None:
        name = 'tmp'
    if save_dir == None:
        save_dir = 'out/gif'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frames = []
    img_list = os.listdir(img_list_dir)
    img_list.sort()
    for img in img_list:
        frames.append(imageio.imread(os.path.join(img_list_dir, img)))

    imageio.mimsave(os.path.join(save_dir, name + '.gif'), frames, 'GIF', fps=fps)



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

    # cv2.destroyAllWindows()
    video.release()



def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img

def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]

def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp


def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    return inp


def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = 0.3
    return mycmap

cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
