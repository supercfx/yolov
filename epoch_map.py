import os
from Map_caculate import Map_caculate
import matplotlib.pyplot as plt
import numpy as np

mAP_savePATH = os.path.join(os.getcwd(), 'results', 'mAP_epoch_figure')
if not os.path.exists(mAP_savePATH):
    os.makedirs(mAP_savePATH)

mAP_list=[]

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

for weight in weights:
    mAP = Map_caculate(weight)
    mAP_list.append(mAP)
    print("%s Conversion completed!" % (weight.split('-')[0]))
max_index = mAP_list.index(max(mAP_list, key=abs))
max_mAP = mAP_list[max_index]
print('The max mAP is in %s epoch' % (max_index + 1))
print('and the value is %s' % max_mAP)
plt.figure()
plt.plot(sort_num, mAP_list, color='blue', linewidth=1.0, linestyle="-", marker='.', )
plt.title("mAP in each epoch")
plt.xlabel("epoch")
plt.ylabel("mAP")
my_y_ticks=np.arange(0,10,1)
plt.savefig('%s/mAP_epoch.jpg' % mAP_savePATH)
plt.show()

