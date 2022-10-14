import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from copy import deepcopy

classes={'0':'other', '1':'xp', '2':'gl'}

def get_imgs(root):
    return [os.path.join(root, x) for x in os.listdir(root) if x.lower().endswith('.jpg') or x.lower().endswith('.png')]

def img_resize(image, width_new = 1280, height_new = 720):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def make_label(imgs, save_path, skip=True):
    if skip and os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf8') as f:
            label_dict = json.load(f)
        label_dict_raw=deepcopy(label_dict)
        skiped=set()
    else:
        label_dict={}

    ptr=-1
    with tqdm(total=len(imgs)) as pbar:
        while ptr<len(imgs)-1:
            pbar.update(1)
            ptr+=1
            p_img = imgs[ptr]
            p_base=os.path.basename(p_img)

            if skip and (p_base not in skiped) and (p_base in label_dict_raw):
                skiped.add(p_base)
                continue

            img=cv2.imread(p_img)
            if img is None:
                continue
            img=img_resize(img)
            cv2.imshow('a', img)
            key = cv2.waitKey()
            if key==ord('a'):
                pbar.update(-2)
                ptr-=2
                print(ptr)
                continue
            elif key==ord('d'):
                continue

            label_dict[p_base]=key-48

            with open(save_path, 'w', encoding='utf8') as f:
                json.dump(label_dict, f, ensure_ascii=False)
    return label_dict

if __name__ == '__main__':
    root='pixiv_crawler/images'

    imgs=get_imgs(root)
    label_dict=make_label(imgs, 'anime_train.json')
    #cv2.imshow('a', np.zeros((2,2)))
    #key = cv2.waitKey()
    #print(key)