#__author__ = "Mohsin Hasan Khan"

#-*- uf8-*-
#base on https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality
# add multiprocessing
from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
from tqdm import tqdm
import zipfile
from IPython.core.display import HTML 
from IPython.display import Image
import multiprocessing
import time
import math
import sys

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(im, flag):
#     path = images_path + img 
#     im = IMG.open(path) #.convert("RGB")
#     im = read_img(img)
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

def average_pixel_width(im): 
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

# %%time
def get_dominant_color(im):
#     path = images_path + img 
#     img = cv2.imread(path)
#     im = im.con
    arr = np.float32(im.convert('RGB').getdata())
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
#     quantized = palette[labels.flatten()]
#     quantized = quantized.reshape((im.size[0],im.size[1],3))

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def get_average_color(img):
    arr = np.float32(img.convert('RGB').getdata())
#     img = arr[:,:3]
#     print(arr.shape)
    average_color = [arr[:, i].mean() for i in range(arr.shape[-1])]
    return average_color


def getSize(filename):
    filename = IMAGE_PATH + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(img):
#     filename = images_path + filename
#     img_size = IMG.open(filename).size
    return img.size

def get_blurrness_score(image):
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.convert('RGB')
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

def process_img_add_featrue(img_name):
    if pd.isna(img_name):
        return np.nan
    try:
        im,size = read_img(img_name)
    except Exception as e:
        print('error',e)
        # raise e
        return np.nan
    dullness = perform_color_analysis(im, 'black')
    whiteness = perform_color_analysis(im, 'white')
    average_pixel_widthv = average_pixel_width(im)
    color = get_dominant_color(im)
    avg_color = get_average_color(im)
    dim = getDimensions(im)
    blurrness = get_blurrness_score(im)
    res = dullness,whiteness,average_pixel_widthv,color, avg_color, dim, blurrness,size
    im.close()
    #print("Done")
    return res

# %%time
import os
pos = 0
plock = multiprocessing.Lock()

def mp_worker(imags):
    global pos
    res = []
    im_ids = imags['image'].values.tolist()
    plock.acquire()
    pos+=1
    t = tqdm(im_ids, position=pos,desc='porc %s'%os.getpid(),file=sys.stdout,mininterval=200)
    
    plock.release()
    for im in t:
        res.append(process_img_add_featrue(im))
    return pd.DataFrame({FEATURE_NAME:res, 'image':im_ids})

def mp_handler(features, df, n_threads, path_str="train"):
    p = multiprocessing.Pool(n_threads)
    batch_size = math.ceil(features.size/n_threads)
    data = []
    print(features.size,batch_size)
    for i in range(n_threads):
        a = features[i*batch_size:(i+1)*batch_size]
        print(a.shape)
        data.append(a)
        
    d = p.map(mp_worker, data)
    res = pd.DataFrame()
    res = res.append(d,ignore_index=True)
    # res = res.reset_index()
    res = post_process_features(res)
    res.to_csv('{}_image_{}.csv'.format(path_str, FEATURE_NAME))
    res1=df.merge(res,how='inner',on='image')
    res1.drop_duplicates(inplace=True)
    res1.to_csv('all_{}_image_{}.csv'.format(path_str, FEATURE_NAME))
    return res1
# def foo(x):
#     print(type(x))
#     print(x)

def post_process_features(df1):
    new_col_list = ['dullness','whiteness','average_pixel_width','dominant_color','avg_color','dimensions','blurrness','size']
    # for n,col in enumerate(new_col_list):
    #     df1[col] = df1[feature_name].apply(lambda x: x[n])
    df1[new_col_list]=df1[FEATURE_NAME].apply(pd.Series)
    # print(df1.columns)
    df1[['dominant_red','dominant_green','dominant_green']] = df1['dominant_color'].apply(pd.Series) / 255
    # df1['dominant_green'] = df1['dominant_color'].apply(lambda x: x[1]) / 255
    # df1['dominant_blue'] = df1['dominant_color'].apply(lambda x: x[2]) / 255
    # df1[['dominant_red', 'dominant_green', 'dominant_blue']].head(5)

    df1[['average_red','average_green','average_blue']] = df1['avg_color'].apply(pd.Series) / 255
    # df1['average_green'] = df1['avg_color'].apply(lambda x: x[1]) / 255
    # df1['average_blue'] = df1['avg_color'].apply(lambda x: x[2]) / 255
    # df1[['average_red', 'average_green', 'average_blue']].head(5)
    df1[['width','height']] = df1['dimensions'].apply(pd.Series)
    # df1['height'] = df1['dimensions'].apply(lambda x : x[1])
    df1 = df1.drop(['dimensions', 'avg_color', 'dominant_color', FEATURE_NAME], axis=1)
    return df1


# In[2]:



IMAGE_PATH = '../input/data/competition_files/train_jpg/'
limit = 100#1503424//2
lock = multiprocessing.RLock()
FEATURE_NAME = 'all'#'dullness'

def read_img(name):
    img_path = os.path.join(IMAGE_PATH, name)
    size = os.path.getsize(img_path)
    return IMG.open(img_path),size
        



# In[6]:


# mp_worker(features['image'])
#if __name__ == '__main__':
    #multiprocessing.freeze_support()  # for Windows support
import time

#train =  pd.read_csv("../input/train.csv",usecols=['image','item_id'], )
test =  pd.read_csv("../input/test.csv",usecols=['image','item_id'], )

#train_images_path = '../input/data/competition_files/train_jpg/'
#train_imgs = os.listdir(train_images_path)

#features_train = pd.DataFrame()
#features_train['image'] = train_imgs
#IMAGE_PATH = train_images_path

#now = time.time()
#res_train = mp_handler(features_train, train, 16, 'train')
#print('time',time.time()-now)

test_images_path = '../input/data/competition_files/test_jpg/'
test_imgs = os.listdir(test_images_path)

features_test = pd.DataFrame()
features_test['image'] = test_imgs
IMAGE_PATH = test_images_path

now = time.time()
res_test = mp_handler(features_test, test, 16, 'test')
print('time',time.time()-now)

