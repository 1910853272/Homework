# coding=utf-8
import scipy.misc, numpy as np, os, sys

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
    #TODO: 使用 scipy.misc 模块读入输入图像 src 并转化成’RGB’ 模式，返回 ndarray 类型数组 img
    img = scipy.misc.imread(src, mode='RGB')
    if img_size:
        img = scipy.misc.imresize(img, img_size)
    return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

