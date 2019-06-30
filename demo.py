from __future__ import division

import sys
import os
import torch as t
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import time
import cv2


LIMIT_MIN_VALUE_AREA = 300
SAVE_FLAG = 0
THRESH = 0.01
IM_RESIZE = False
tracker = None
leftstart = 440  #左边
rightend = 580  #右边
top = 170   #顶部
bottom = 300    #底部
ori_width = 1000
ori_height = 563

def transfer_matrix(box):
    return (box[0],box[1],box[2]+box[0],box[3] + box[1])

#判断是否在感兴趣区域
def out_of_area(box):
    if box[1] + box[3] < top:
        return True
    if box[0] > rightend:
        return True
    if box[0] + box[2] < leftstart:
        return True
    if box[1] > bottom:
        return True
    return False

def calcRectanctArea(box1):
    #xmin,ymin,xmax,ymax
    return abs(box1[2] - box1[0]) * abs(box1[1] - box1[3])

#计算两个矩阵是否相交
def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False

#计算矩阵重合度
def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    #if mat_inter(box1, box2) == True:
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    col = abs(min(x02, x12) - max(x01, x11))
    row = abs(min(y02, y12) - max(y01, y11))
    intersection = col * row
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)
    coincide = intersection / (area1 + area2 - intersection)
    return coincide
    # else:
    #     return False

#判断box1是否包含box2
def isIncludeTheMatrix(box1,box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    if x11 >= x01:
        if y11 >= y01:
            if x02 >= x12:
                if y02 >= y12:
                    return True
    return False

#读取图片
def read_img(frame):
    # f = Image.open(path)
    # if IM_RESIZE:
    #     f = f.resize((640,480), Image.ANTIALIAS)
    # f.convert('RGB')
    f = Image.fromarray(frame).convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    _, H, W = img.shape
    img = img.transpose((2,0,1))
    img = preprocess(img)
    _, o_H, o_W = img.shape
    scale = o_H / H
    f = f.resize((img.shape[2],img.shape[1]), Image.ANTIALIAS)
    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    # print(type(img_raw_final))
    # print(img_raw_final.shape)
    return img, img_raw_final, scale

#检测
def detect(head_detector,trainer,frame):
    #file_id = utils.get_file_id(img_path)
    img, img_raw, scale = read_img(frame)
    img = at.totensor(img)
    img = img[None, : ,: ,:]
    img = img.cuda().float()
    # st = time.time()
    pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
    # et = time.time()
    # tt = et - st
    # print ("[INFO] Head detection over. Time taken: {:.4f} s".format(tt))
    matrixs = []        #存放一张图里面的所有人头矩形
    need_delete = set()    #需要被删除的矩形
    height = np.shape(img_raw)[0]
    width = np.shape(img_raw)[1]


    #绘制感兴趣区域
    # cv2.line(img_raw, (leftstart, top), (rightend, top), (0,0,255),2)
    # cv2.line(img_raw, (leftstart, bottom), (rightend, bottom), (0,0,255),2)
    # cv2.line(img_raw, (leftstart, top), (leftstart, bottom), (0, 0, 255),2)
    # cv2.line(img_raw, (rightend, top), (rightend, bottom), (0, 0, 255),2)

    #添加所有的矩形区域到list
    for i in range(pred_bboxes_.shape[0]):
        ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
        matrixs.append([xmin,ymin,xmax,ymax])

    #计算重叠的矩形，并标记需要删除的矩形
    for i in np.arange(len(matrixs)):
        if(calcRectanctArea(matrixs[i])<=LIMIT_MIN_VALUE_AREA):
            need_delete.add(i)
        for j in np.arange(len(matrixs)):
            if(i != j):
                if(mat_inter(matrixs[i],matrixs[j])):
                    if(calcRectanctArea(matrixs[i])>calcRectanctArea(matrixs[j])):
                        need_delete.add(j)
                    else:
                        need_delete.add(i)

    deleted_matrix = []
    for i in np.arange(len(matrixs)):
        if(i not in need_delete):
            deleted_matrix.append(matrixs[i])
    del(matrixs)    #释放空间
    # 划分区域
    return img_raw,deleted_matrix

if __name__ == '__main__' :
    model_path = './checkpoints/head_detector_final'
    video_path = './data/acvis09.avi'
    
    #设置使用显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    #提取特征
    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2, 4])

    #训练器
    trainer = Head_Detector_Trainer(head_detector).cuda()
    
    print("loading the model...")
    #加载模型
    trainer.load(model_path)
    print('detection...')

    camera = cv2.VideoCapture(video_path)
    trackerlist = []
    tracker = cv2.MultiTracker_create()
    init_once = False
    c = 0
    p = 15      #设置间隔的帧数进行检测
    tracker_amount = 0
    tracker_box_list = []
    boxes = None
    tracker_list = []
    person_same_rate = 0.25     #面积超过0.25作为同一个人处理
    count_people = 0
    while camera.isOpened():
        ok, image = camera.read()
        image = cv2.resize(image, (ori_width, ori_height), cv2.INTER_LINEAR)
        c += 1

        if c%150 == 0:
            tracker_amount = 0
            tracker_list = []
            tracker_box_list = []
            tracker = cv2.MultiTracker_create()
            boxes = None


        #追踪器大于1的情况
        if tracker_amount > 0:
            ok, boxes = tracker.update(image)
            del_boxes = []
            #在感兴趣区域以外则删除该追踪器
            for indx in np.arange(len(boxes)):
                temp_position = (boxes[indx][0],boxes[indx][1],boxes[indx][2],boxes[indx][3])
                if(out_of_area(temp_position)):
                    del_boxes.append(indx)  #添加删除序号
                    tracker_amount -= 1
                    count_people+=1
                    print(count_people)
                    continue
                for indx_compare in np.arange(indx+1,len(boxes)):
                    #是否相交
                    if mat_inter(transfer_matrix(boxes[indx]),transfer_matrix(boxes[indx_compare])):
                        #是否超过面积相交范围
                        if solve_coincide(transfer_matrix(boxes[indx]),transfer_matrix(boxes[indx_compare])) >= person_same_rate:
                            del_boxes.append(indx_compare)
                            tracker_amount -= 1

            if(len(del_boxes)>0):
            #获取追踪器位置
                objs = tracker.getObjects()

                tracker = cv2.MultiTracker_create()
                for indx in np.arange(len(objs)):
                    if(indx not in del_boxes):
                        tracker.add(cv2.TrackerMIL_create(),image,tuple(objs[indx]))

            #清除追踪器具体位置信息
            tracker_box_list.clear();

            # 0代表 xmin,1代表 ymin,2代表dx,3代表dy
            #绘制检测到的追踪器矩形框
            for indx in np.arange(len(boxes)):
                if indx not in del_boxes:
                    tracker_box_list.append(boxes[indx])
                    p1 = (int(boxes[indx][0]), int(boxes[indx][1]))
                    p2 = (int(boxes[indx][0] + boxes[indx][2]), int(boxes[indx][1] + boxes[indx][3]))
                    cv2.rectangle(image, p1, p2, (0, 255, 0))


        #隔p帧显示一次
        if c % p == 0:
            image,deleted_matrix = detect(head_detector, trainer, image)
                #ymin, xmin, ymax, xmax
                # 0代表 xmin,1代表 ymin,2代表xmax,3代表ymax
            for i in np.arange(len(deleted_matrix)):
                deltay = deleted_matrix[i][3] - deleted_matrix[i][1]
                deltax = deleted_matrix[i][2] - deleted_matrix[i][0]
                #box1 include box2
                if isIncludeTheMatrix([leftstart, top, rightend, bottom], deleted_matrix[i]):
                    temp_box = (deleted_matrix[i][0], deleted_matrix[i][1], deltax, deltay)
                    if init_once == False:
                        t1 = cv2.TrackerMIL_create()
                        tracker_list.append(t1)
                        ok = tracker.add(t1,image,(deleted_matrix[i][0], deleted_matrix[i][1], deltax, deltay))
                        cv2.rectangle(image, (deleted_matrix[i][0], deleted_matrix[i][1]),(deleted_matrix[i][2],deleted_matrix[i][3]), (0, 255, 0))
                        tracker_amount += 1
                        tracker_box_list.append((deleted_matrix[i][0], deleted_matrix[i][1], deleted_matrix[i][2], deleted_matrix[i][3]))
                        init_once = True

                    flag = True
                    t_box = (deleted_matrix[i][0], deleted_matrix[i][1], deleted_matrix[i][2], deleted_matrix[i][3])
                    for indx in np.arange(len(tracker_box_list)):
                        if(mat_inter(tracker_box_list[indx],t_box)):
                            #print("相交")
                            if(solve_coincide(tracker_box_list[indx],t_box)>0.2):
                                flag = False
                                break
                    #print(flag)
                    if flag == True:
                        t1 = cv2.TrackerMIL_create();
                        tracker_list.append(t1);
                        ok = tracker.add(t1, image, temp_box)
                        tracker_box_list.append(temp_box)
                        tracker_amount += 1

                        # ymin,xmin,ymax,xmax
                    #utils.draw_bounding_box_on_image_array(image, deleted_matrix[i][1], deleted_matrix[i][0],deleted_matrix[i][3], deleted_matrix[i][2])
        cv2.imshow("predict", image)
        k = cv2.waitKey(1)
        if k == 27: break  # esc pressed
    camera.release()
    cv2.destroyAllWindows()







