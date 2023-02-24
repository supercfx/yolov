import matplotlib.pyplot as plt
import numpy as np
# i=1
# f=open('/media/lab206/9E4ADB134ADAE755/YOLOV4-tiny/testplt.txt','a')
# while i <=10:
#     f.write("{}_{}\n".format(i,i+3))
#     i=i+1
# f.close()
f=open('/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/iter_loss.txt','r')
line = f.readlines()
x=[]
y=[]
for li in line:
    li = li[:-1]
    x.append(int(li.split('_')[0]))
    y.append(float(li.split('_')[1]))
plt.figure()
plt.plot(x[30:], y[30:], color='blue', linewidth=0.4, linestyle="-", )
plt.title("Train_loss")
plt.xlabel("iters")
plt.ylabel("loss")

# my_x_ticks=np.arange(2000,22000,2000)
# plt.axis([0,22000])
xtips=np.arange(0,22000,2000)
my_x_ticks=["0","2k","4k","6k","8k","10k","12k","14k","16k","18k","20k"]
my_y_ticks=np.arange(0,200,20)
plt.xticks(xtips,my_x_ticks)
plt.yticks(my_y_ticks)
# plt.xlim(0,22000)
# plt.ylim(0,200)
plt.savefig('/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/iter_loss.jpg')

plt.show()
