#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 07:01:30 2018

@author: mohsin
"""

#__author__ = "Mohsin Hasan Khan"

#Extract image features based on this paper - https://storage.googleapis.com/kaggle-forum-message-attachments/328059/9411/dimitri-clickadvert.pdf
import time
import numpy as np
import pandas as pd 
import cv2
import os 
from tqdm import tqdm
tqdm.pandas(tqdm)
from multiprocessing import Pool
import dask.dataframe as dd
from dask.multiprocessing import get
from PIL import Image
#%%
def img_stats(img):
    img_arr = np.array(Image.open(img))
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
    img_hsv_v = img_hsv[:,:,2]
    br_mean, br_std, br_min = np.mean(img_hsv_v), np.std(img_hsv_v), np.min(img_hsv_v)
    
    img_hsv_s = img_hsv[:,:,1]
    sat_avg, sat_std = np.mean(img_hsv_s), np.std(img_hsv_s)
    
    img_yuv = cv2.cvtColor(img_arr, cv2.COLOR_BGR2YUV)
    img_yuv_y = img_yuv[:,:,0]
    lum_mean, lum_std, lum_min, lum_max = np.mean(img_yuv_y), np.std(img_yuv_y), np.min(img_yuv_y), np.max(img_yuv_y)
    
    contrast = np.std((img_yuv_y - lum_min)/(lum_max - lum_min))

    rg = img_arr[:,:,2] - img_arr[:,:,1] 
    yb = 1/2*(img_arr[:,:,2] + img_arr[:,:,1] ) - img_arr[:,:,0]
    CF = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3*np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    
    orb = cv2.ORB_create()
    #fast = cv2.FastFeatureDetector_create()
    kp = len(orb.detect(img_arr,None))
    #kp = len(fast.detect(img_arr,None))
    
    all_hist = cv2.calcHist([img_arr],[0, 1, 2],None,[8,8,8],[0,256,0,256,0,256]).flatten()
    dominant_color = np.argmax(all_hist)
    dominant_color_ratio = all_hist[dominant_color]/sum(all_hist)
    simplicity = len(all_hist[all_hist/sum(all_hist) > 0.03])
    
    img_arr_small = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr_small = cv2.resize(img_arr_small, (64, 64))
    c = cv2.dft(np.float32(img_arr_small), flags = cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(c[:,:,0], c[:,:,1])
    logmag = np.log(mag)
    smooth = cv2.boxFilter(logmag, -1, (3,3))
    spectralResidual = logmag - smooth
    spectralResidual = np.exp(spectralResidual)
    
    c[:,:,0] = c[:,:,0] * spectralResidual / mag
    c[:,:,1] = c[:,:,1] * spectralResidual / mag
    c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = cv2.magnitude(c[:,:,0], c[:,:,1])
    mag = mag**2
    cv2.normalize(cv2.GaussianBlur(mag,(5,5),8,8), mag, 0., 1., cv2.NORM_MINMAX)
    _, saliency = cv2.threshold(np.uint8(mag*255), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    object_ratio = sum(saliency.flatten())/len(saliency.flatten())
    return np.array([br_mean, br_std, br_min, sat_avg, sat_std, lum_mean, lum_std, lum_min, contrast, CF, kp,
            dominant_color, dominant_color_ratio, simplicity, object_ratio])


def process_img(imgfile):
    try:
        features = img_stats(imgfile)
    except:
        print("Exception occured")
        features = np.array([-1]*15)
    return features

#%%
if __name__ == "__main__":
    
    train_images_path = '../input/data/competition_files/train_jpg/'
    test_images_path = '../input/data/competition_files/test_jpg/'
    train_imgs = os.listdir(train_images_path)
    train_imgs = [os.path.join(train_images_path, imgf) for imgf in train_imgs]
    test_imgs = os.listdir(test_images_path)
    test_imgs = [os.path.join(test_images_path, imgf) for imgf in test_imgs]
    
    all_imgs = train_imgs + test_imgs
    df = pd.DataFrame()
    df['image_path'] = all_imgs
    df['image'] = df['image_path'].str.split('/').str.get(-1).str.split('.').str.get(0)
    
    start_time = time.time()
    #print(start_time)
    ddata = dd.from_pandas(df['image_path'].astype(str), npartitions=15)
    all_features = ddata.map_partitions(
                        lambda tmp: tmp.progress_apply(
                                lambda x: process_img(x))).compute(get=get)
    
    print((time.time() - start_time))
    all_features = np.vstack(all_features)
    
    fnames = ['br_mean', 'br_std', 'br_min', 'sat_avg', 'sat_std', 'lum_mean', 'lum_std', 'lum_min', 'contrast', 
              'CF', 'kp', 'dominant_color', 'dominant_color_ratio', 'simplicity', 'object_ratio']
    all_features = pd.DataFrame(all_features, columns=fnames)
    df_all = pd.concat([df, all_features], axis=1)
    
    print(df_all.head())
    df_all.to_csv("../utility/df_image_feats3.csv", index=False)
    #batch_size = 10000
    #all_feats = []
    #n_batches = int(np.ceil(len(df)/batch_size))
    #for i in range(n_batches):
    #    with Pool(16) as pool:
    #        print("Processing batch {}".format(i))
    #        start_idx = int(i*batch_size)
    #        end_idx  = int((i+1)*batch_size)
    #        results = [pool.apply_async(process_img, args=(x,)) for x in df['image_path'].iloc[start_idx: end_idx].tolist()]
    #        output = [p.get(timeout=10) for p in results]
            
    #        output = np.array(output)
    #        np.save("image_feats3_batch_{}.npy".format(i), output)
    #        all_feats.append(output)
            