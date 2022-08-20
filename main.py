#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/17 16:58 
"""
import io
import os
import uuid
import datetime
import cv2
import torch
import yaml
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from minio import Minio

from database.mysqldb.mysqldb import MySqlDB
import numpy as np

from entity.vehicle_capture import VehicleCapture
from model.lpr.STN.model.STN import STNet
from model.lpr.model import build_lprnet

app = FastAPI(title="车牌识别接口 包括yolo车牌检测 + 车牌号识别")
yaml_file = 'config/application.yaml'


def load_cfg():
    with open(yaml_file, encoding="utf-8") as f:
        case_data = yaml.safe_load(f.read())
        return case_data


cfg = load_cfg()
print(cfg)
database_cfg = cfg['database']
mysql = MySqlDB(database_cfg['host'], database_cfg['username'], database_cfg['password'], database_cfg['port'], database_cfg['database'])
minio_cfg = cfg['minio']
minio_client = Minio(minio_cfg['host'], access_key=minio_cfg['access_key'], secret_key=minio_cfg['secret_key'], secure=False)
bucket_name = minio_cfg['bucket_name']
minio_url_head = minio_cfg['host'] + '/' + bucket_name + '/'

# load yolo model
# Initialization parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

modelConfiguration = "model/yolo/darknet-yolov3.cfg"
modelWeights = "weights/model.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]
# LPR_NET init
LPR_Net = build_lprnet(lpr_max_len=8, phase='test', class_num=len(CHARS))
LPR_Net = LPR_Net.cuda()
LPR_Net.load_state_dict(torch.load('weights/Final_LPRNet_model.pth'))
LPR_Net.eval()
STN = STNet()
STN = STN.cuda()
STN.eval()
STN.load_state_dict(torch.load('weights/STN_Model_LJK_CA_XZH.pth', map_location=lambda storage, loc: storage))
print("Successful to build network!")


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def crop_and_save_to_minio(left, top, right, bottom, filename, im_height, im_width, im, index):
    left = max(0, left)
    top = max(0, top)
    bottom = min(bottom, im_height-1)
    right = min(right, im_width-1)
    im = im[top:bottom, left:right, :]
    filename = str(uuid.uuid1()) + '.jpg'
    bs = cv2.imencode(".jpg", im)[1].tobytes()
    content = io.BytesIO(bs)
    minio_client.put_object(bucket_name=bucket_name, object_name=filename, data=content, length=len(bs))
    return im, filename


def postprocess(frame, outs, filename=None):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print(detection[4], " - ", scores[classId],
                      " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    images = []
    filenames = []
    for index, i in enumerate(indices):
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        image, filename = crop_and_save_to_minio(left, top, left + width, top + height, filename, frameHeight, frameWidth, frame, index)
        images.append(image)
        filenames.append(filename)

    return images, filenames


def yolo_lpr_detection(content):
    im_pil = Image.open(content)
    frame = cv2.cvtColor(np.asarray(im_pil), cv2.COLOR_RGB2BGR)
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    images, filenames = postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the
    # timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    return images, filenames


def saveResult(ve_image_url, lp_images):
    lp_images = [minio_url_head + image for image in lp_images]
    lp_image_url = ",".join(lp_images)
    vehicle_capture = VehicleCapture()
    vehicle_capture.vehicle_image_url = ve_image_url
    vehicle_capture.lp_url = lp_image_url
    vehicle_capture.location = "location"
    vehicle_capture.capture_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vehicle_capture.create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vehicle_capture.update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    session = mysql.session()
    session.add(vehicle_capture)

    session.commit()
    session.close()


@app.post("/image/upload", description='单张图片上传')
async def uploadFile(file: UploadFile = File(...)):
    filename = file.filename
    suffix = filename.split(".")[-1]
    new_name = str(uuid.uuid1()).replace('-', '')
    new_name = new_name + "." + suffix
    contents = await file.read()
    content = io.BytesIO(contents)
    minio_client.put_object(bucket_name=bucket_name, object_name=new_name, data=content, length=len(contents))
    ve_image_url = minio_url_head + new_name
    # lp detection
    images, lp_images = yolo_lpr_detection(content)
    saveResult(ve_image_url, lp_images)
    # lp recognition


    return {'code': 200, 'success': True, 'message': '操作成功'}


