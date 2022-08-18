#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/17 16:58 
"""
import io
import os
import uuid

import cv2
import yaml
from fastapi import FastAPI, UploadFile, File
from minio import Minio

from database.mysqldb.mysqldb import MySqlDB

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
minio_client = Minio('127.0.0.1:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
minio_cfg = cfg['minio']
bucket_name = minio_cfg['bucket_name']

# load yolo model
# Initialization parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

modelConfiguration = "darknet-yolov3.cfg"
modelWeights = "model.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


@app.post("/image/upload", description='单张图片上传')
async def uploadFile(file: UploadFile = File(...)):
    filename = file.filename
    suffix = filename.split(".")[-1]
    new_name = uuid.uuid1()
    print(type(new_name))
    contents = await file.read()
    # print(contents)
    content = io.BytesIO(contents)
    print(content)
    # minio_client.fget_object(bucket_name, 'testfile', contents, request_headers=None)
    # file = os.stat(contents)
    # res = minio_client.put_object(bucket_name=bucket_name, object_name='testfile', data=content, length=len(contents))
    # url = minio_client.get_presigned_url("GET", bucket_name, "testfile")
    # print(url)
    return "123"
    # content = io.BytesIO(contents)

'''
print(mysql)
vehicle = VehicleCapture()
vehicle.number = "123456"
vehicle.vehicle_image_url = "vehicle_image_url"
vehicle.lp_url = "lp_url"
vehicle.location = "location"
vehicle.capture_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
vehicle.create_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
vehicle.update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

session = mysql.session()
session.add(vehicle)

session.commit()
session.close()
'''

