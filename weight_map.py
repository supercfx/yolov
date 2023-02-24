import os
import glob
import json
import shutil
import operator
import sys
import argparse
import math

import numpy as np
from get_dr_txt import mAP_Yolo
from PIL import Image
from Map_caculate import Map_caculate
import matplotlib.pyplot as plt
from tqdm import tqdm

weight_filepath = r'/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/traffic-yolov4-tiny-pytorch/logs'
filelists = os.listdir(weight_filepath)
sort_num = []
for file in filelists:
    sort_num.append(int((file.split("-")[0])[5:]))
sort_num.sort()
weights = []
for num in sort_num:
    for fil in filelists:
        if str(num) == (fil.split("-")[0])[5:]:
            weights.append(fil)


image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

for weight in weights:
    if not os.path.exists("./input/%s"%(weight.split('-')[0])):
        os.makedirs("./input/%s"%(weight.split('-')[0]))
    if not os.path.exists("./input/%s/detection-results"%(weight.split('-')[0])):
        os.makedirs("./input/%s/detection-results"%(weight.split('-')[0]))
    #if not os.path.exists("./input/%s/images-optional"%(weight.split('-')[0])):
        #os.makedirs("./input/%s/images-optional"%(weight.split('-')[0]))
for weight in weights:
    detect = mAP_Yolo(model_path='logs/%s' % weight)
#
    for image_id in tqdm(image_ids):
        image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
        image = Image.open(image_path)
        # 开启后在之后计算mAP可以可视化
        # image.save("./input/images-optional/"+image_id+".jpg")
        detect.detect_image(image_id,image)
    print("%s Conversion completed!"%(weight.split('-')[0]))
