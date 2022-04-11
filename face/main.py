from tkinter import Image
from face.yolov4.facedet import FaceDetector

from flask import Flask, jsonify, request

import base64
import cv2
import numpy as np

detector = FaceDetector()


def img_to_base64(img_array):
    # 传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) #RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1] #用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes() #转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
    return base64_str
    
def base64_to_img(base64_str):
    # 传入为RGB格式下的base64，传出为RGB格式的numpy矩阵
    byte_data = base64.b64decode(base64_str)#将base64转换为二进制
    encode_image = np.asarray(bytearray(byte_data), dtype="uint8")# 二进制转换为一维数组
    img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)# 用cv2解码为三通道矩阵
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)# BGR2RGB
    return img_array

app = Flask(__name__)                       # 创建一个服务，赋值给APP
@app.route('/detect', methods=['post'])     # 指定接口访问的路径，支持什么请求方式get，post

# key_values方式传参
def detect():
    img_str = str(request.form.get('img'))      # 获取接口请求中form-data的img参数传入的值
    img = base64_to_img(img_str)
    
    state = 0

    pred = detector.detect(img)
    # 检测
    if pred is not None:
        # 循环处理识别结果
        for x1, y1, x2, y2, p, cls_id in pred:
            img = detector.detect_mark(img)
        state = 1

    rlt = {'img':img_to_base64(img), 'state':state}

    return jsonify(rlt)

#这个host：windows就一个网卡，可以不写，而liux有多个网卡，写成0:0:0可以接受任意网卡信息
app.run(host='127.0.0.1', port=8088, debug=True)

