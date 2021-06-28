#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from natsort import natsorted
from glob import glob
from shutil import copyfile

import argparse
parser = argparse.ArgumentParser(description='Splitting the Images')

parser.add_argument('--src', default='./dataset/iSAID', type=str, help='path for the original dataset')
parser.add_argument('--tar', default='./dataset/iSAID_patches', type=str, help='path for saving the new dataset')
parser.add_argument('--image_sub_folder', default='images', type=str, help='name of subfolder inside the training, validation and test folders')
parser.add_argument('--set', default="train,val,test", type=str)
parser.add_argument('--patch_width', default=800, type=int, help='Width of the cropped image patch')
parser.add_argument('--patch_height', default=800, type=int, help='Height of the cropped image patch')
parser.add_argument('--overlap_area', default=200, type=int, help='Overlap area')


args = parser.parse_args()

src=args.src
tar=args.tar
modes1=args.set.split(',')
mode2=args.image_sub_folder
patch_H, patch_W = args.patch_width, args.patch_height # image patch width and height
overlap = args.overlap_area #overlap area
extras = []

for i in modes1:
    if i == 'train' or i == 'val':
        extras=['',  '_instance_color_RGB',]
    elif i == 'test':
        extras = ['']
    else:
        print('Invalid input')

for mode1 in modes1:
    src_path = src  + "/" + mode1 + "/" + mode2
    tar_path = tar  + "/" + mode1 + "/" + mode2

    os.makedirs(tar_path, exist_ok=True)

    files = glob(src_path + "/*.png")
    files = [os.path.split(i)[-1].split('.')[0] for i in files if '_' not in os.path.split(i)[-1]]
    files = natsorted(files)

    for file_ in files:
        if file_ == 'P1527' or file_ == 'P1530':
            continue
        for extra in extras:
            filename = file_ + extra + '.png'
            full_filename = src_path + '/' + filename
            img = cv2.imread(full_filename)
            img_H, img_W, _ = img.shape
            X = np.zeros_like(img,dtype=float)
            h_X, w_X,_ = X.shape

            if extra == '_instance_color_RGB':
                save_path = tar_path + '/masks'
                os.makedirs(save_path, exist_ok=True)
            else:
                save_path = tar_path + '/images'
                os.makedirs(save_path, exist_ok=True)

            if img_H > patch_H and img_W > patch_W:
                for x in range(0, img_W, patch_W-overlap):
                    for y in range(0, img_H, patch_H-overlap):
                        x_str = x
                        x_end = x + patch_W
                        if x_end > img_W:
                            diff_x = x_end - img_W
                            x_str-=diff_x
                            x_end = img_W
                        y_str = y
                        y_end = y + patch_H
                        if y_end > img_H:
                            diff_y = y_end - img_H
                            y_str-=diff_y
                            y_end = img_H
                        if extra == '_instance_color_RGB':
                            patch = img[y_str:y_end, x_str:x_end]
                        else:
                            patch = img[y_str:y_end,x_str:x_end,:]
                        image = file_+'_'+str(y_str)+'_'+str(y_end)+'_'+str(x_str)+'_'+str(x_end)+extra+'.png'
                        print(image)
                        save_path_image = save_path + '/' + image
                        # if extra == '_instance_color_RGB':
                        #     patch = Image.fromarray(patch)
                        #     patch = patch.convert('L')
                        #     patch.save(save_path_image)
                        # else:
                        cv2.imwrite(save_path_image, patch)
            else:
                copyfile(full_filename, save_path+'/'+filename)
