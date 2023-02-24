import matplotlib.pyplot as plt
import numpy as np
# i=1
# f=open('/media/lab206/9E4ADB134ADAE755/YOLOV4-tiny/testplt.txt','a')
# while i <=10:
#     f.write("{}_{}\n".format(i,i+3))
#     i=i+1
# f.close()
f=open('/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/iter_loss_val.txt','r')
line = f.readlines()
x=[]
y=[]
for li in line:
    li = li[:-1]
    x.append(int(li.split('_')[0]))
    y.append(float(li.split('_')[1]))
xtips=np.arange(0,2200,150)
plt.figure()
plt.plot(x, y, color='blue', linewidth=0.8, linestyle="-", )
plt.title("Val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")

# my_x_ticks=np.arange(2000,22000,2000)
# plt.axis([0,22000])

my_x_ticks=["0","10","20","30","40","50","60","70","80","90","100","110","120","130","140","150"]
my_y_ticks=np.arange(0,120,20)
plt.xticks(xtips,my_x_ticks)
plt.yticks(my_y_ticks)
# plt.xlim(0,22000)
# plt.ylim(0,120)
plt.savefig('/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/iter_loss_val.jpg')

plt.show()
