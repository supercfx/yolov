#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from lxml import etree
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import click

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

#pascalVocReader readers the voc xml files parse it
class PascalVocReader:
    """
    this class will be used to get transfered width and height from voc xml files
    """
    def __init__(self, filepath,width,height):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        self.width=width
        self.height=height

        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, bndbox, width,height):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        width_trans = (xmax - xmin)/width*self.width
        height_trans = (ymax-ymin)/height *self.height
        if width_trans > 0 and height_trans > 0:
            points = [width_trans,height_trans]
            self.shapes.append((points))
        # points = [width_trans, height_trans]
        # self.shapes.append((points))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        pic_size = xmltree.find('size')
        size = (int(pic_size.find('width').text),int(pic_size.find('height').text))
        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            self.addShape(bndbox, *size)
        return True

class create_w_h_txt:
    def __init__(self,vocxml_path,width_hight,txt_path):
        self.voc_path = vocxml_path
        self.txt_path = txt_path
        self.width_hight = width_hight
    def _gether_w_h(self):
        pass
    def _write_to_txt(self):
        pass
    def process_file(self):
        file_w = open(self.txt_path,'w')
       # print (self.txt_path)
        for file in os.listdir(self.voc_path):
            file_path = os.path.join(self.voc_path, file)
            xml_parse = PascalVocReader(file_path,self.width_hight[0],self.width_hight[1])
            data = xml_parse.getShapes()
            for w,h in data :
                txtstr = str(w)+' '+str(h)+'\n'
                #print (txtstr)
                file_w.write(txtstr)
        file_w.close()

class kMean_parse:
    def __init__(self,n_clusters,path_txt):
        self.n_clusters = n_clusters
        self.path = path_txt
        self.km = KMeans(n_clusters=self.n_clusters,init="k-means++",n_init=10,max_iter=3000000,tol=1e-3,random_state=0)
        self._load_data()

    def _load_data (self):
        self.data = np.loadtxt(self.path)

    def parse_data (self):
        self.y_k = self.km.fit_predict(self.data)
        print(self.km.cluster_centers_)

    def plot_data (self):

        cValue = ['orange','r','y','green','b','gray','black','purple','brown','tan']

        for i in range(self.n_clusters):
            plt.scatter(self.data[self.y_k == i, 0]/2, self.data[self.y_k == i, 1]/3, s=10, c=cValue[i%len(cValue)], marker="o",
                        label="cluster "+str(i))
            # plt.scatter(self.data[self.y_k == i, 0], self.data[self.y_k == i, 1], s=20, c=cValue[i % len(cValue)],
            #             marker="o")

       # draw the centers
        plt.scatter(self.km.cluster_centers_[:, 0]/2, self.km.cluster_centers_[:, 1]/3, s=50, marker="*", c="purple", label="cluster center")
        plt.legend()
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid()
        plt.savefig("kmeans.results.jpg")
        plt.show()

@click.command()
@click.option('--xml_path', default='/home/susiyu/Desktop/YOLOV4-tiny/YOLOV4-tiny/traffic-yolov4-tiny-pytorch/VOCdevkit/VOC2007/Annotations', help='path of xml label')
@click.option('--width_hight', default=[608,608], help='width and hight of training input')
@click.option('--n_clusters', default=6, help='number of clusters')
def get_anchors(xml_path,width_hight,n_clusters):
    whtxt = create_w_h_txt(xml_path,width_hight,"./data1.txt") #指定为voc标注路径，以及存放生成文件路径
    whtxt.process_file()
    kmean_parse = kMean_parse(n_clusters,"./data1.txt")
    kmean_parse.parse_data()
    kmean_parse.plot_data() # 图示


if __name__ == '__main__':
    get_anchors()