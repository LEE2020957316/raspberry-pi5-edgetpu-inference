#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera & GUI process (Python 3.11)
* Captures PiCamera2 RGB frames
* Publishes JPEG via ZeroMQ PUB tcp://127.0.0.1:5555
* Receives detection/emotion JSON via SUB tcp://127.0.0.1:5556
"""
import os, psutil
import cv2, zmq, time
import numpy as np
from picamera2 import Picamera2
 
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB);  pub.bind('tcp://127.0.0.1:5555')
sub = ctx.socket(zmq.SUB);  sub.connect('tcp://127.0.0.1:5556'); sub.setsockopt(zmq.SUBSCRIBE, b'')
 
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={'format': 'RGB888', 'size': (640, 480)}  # ✅ 改为 640×480
    )
)
picam2.start(); time.sleep(0.2)

process = psutil.Process(os.getpid())
process.cpu_percent(interval=None)  # 第一次调用预热
 
print('📷 cam_gui.py running – ESC to quit')
while True:
    start = time.time()
    ##frame = picam2.capture_array()              # RGB888 H×W×3
    ##frame = cv2.imread('picture1234/image00.png')
    ##frame = cv2.imread('picture1234/image01.png')
    ##frame = cv2.imread('picture1234/image02.png')
    frame = cv2.imread('picture1234/image03.png')
    ret, buf = cv2.imencode('.png', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    time.sleep(0.08)
    pub.send(buf)
    print(f'Sent {len(buf)} bytes')                               # → TPU 进程
 
    try:
        msg = sub.recv_json(flags=zmq.NOBLOCK)  # ← 结果
        for obj in msg['data']:
            x0,y0,x1,y1 = obj['box']
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)
            cv2.putText(frame,obj['emo'],(x0,y0-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    except zmq.Again:
        pass
 
    cv2.imshow('PiCam + Emotion', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    cpu_pct = process.cpu_percent(interval=None)
    duration = (time.time() - start) * 1000
    print(f"[INFO] 当前帧耗时: {duration:.1f} ms | CPU 占用率: {cpu_pct:.1f}%")
    
picam2.stop(); cv2.destroyAllWindows()
