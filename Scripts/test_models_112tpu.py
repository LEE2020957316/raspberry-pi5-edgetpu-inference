#!/usr/bin/env python3
"""
Benchmark pipeline – prints per-frame det/fer FPS
and per-window (50 帧) overall FPS.
"""
import os, psutil  # ← 新增
import time, datetime, zmq, cv2, numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, input_tensor, output_tensor

USE_TPU_FACE = True
USE_TPU_FER  = True
FPS_WINDOW   = 50                     # <<< 整管线统计窗口

DIR = '/home/pi/Desktop/models1/face1'
FACE_TPU = f'{DIR}/face_detector_post_edgetpu.tflite'
FACE_CPU = f'{DIR}/face_detector_cpu.tflite'
FER_TPU  = f'{DIR}/facial_expression_recognition_mobilefacenet_2022july_full_integer_quant_edgetpu.tflite'
FER_CPU  = f'{DIR}/facial_expression_recognition_mobilefacenet_2022july_full_integer_quant.tflite'

DET_PATH = FACE_TPU if USE_TPU_FACE else FACE_CPU
FER_PATH = FER_TPU  if USE_TPU_FER  else FER_CPU

det = make_interpreter(DET_PATH, device='usb' if USE_TPU_FACE else '')
det.allocate_tensors()
fer = make_interpreter(FER_PATH, device='usb' if USE_TPU_FER  else '')
fer.allocate_tensors()

DW, DH = input_size(det)
EMO = ['Neutral','Happy','Sad','Surprise','Anger','Disgust','Fear']

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB); sub.connect('tcp://127.0.0.1:5555'); sub.setsockopt(zmq.SUBSCRIBE,b'')
pub = ctx.socket(zmq.PUB); pub.bind('tcp://127.0.0.1:5556')

print(f"🚀 ready  | Face TPU={USE_TPU_FACE}  FER TPU={USE_TPU_FER}")

# ── 整管线 FPS 计数器 ───────────────────────────
frame_cnt = 0                         # <<< 新增
t_start   = time.time()               # <<< 新增

cpu_history = []
mem_history = []  

while True:
    jpg = sub.recv()
    img = cv2.imdecode(np.frombuffer(jpg,np.uint8), cv2.IMREAD_COLOR)
    if img is None: continue
    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── det ───────────────────────────────
    det_in = cv2.resize(rgb,(DW,DH))
    input_tensor(det)[...] = det_in
    t0 = time.time(); det.invoke(); t1 = time.time()
    det_fps = 1/(t1-t0)

    boxes  = output_tensor(det,0)[0].copy()
    scores = output_tensor(det,2)[0].copy()
    num_det = int(output_tensor(det,3)[0])

    # ── fer ───────────────────────────────
    fer_times = []
    data = []
    for i in range(num_det):
        if scores[i] < .25: continue
        ymin,xmin,ymax,xmax = boxes[i]
        x0,y0,x1,y1 = map(int,[xmin*w,ymin*h,xmax*w,ymax*h])
        face = rgb[y0:y1,x0:x1]
        if face.size==0: continue
        fer_in = cv2.resize(face,(112,112))
        input_tensor(fer)[...] = fer_in
        t2=time.time(); fer.invoke(); t3=time.time()
        fer_times.append(t3-t2)
        ##emo = EMO[int(np.argmax(output_tensor(fer,0)[0]))]
        ##print("Raw output:", output_tensor(fer, 0)[0])
        raw = output_tensor(fer, 0)[0].copy()
        scale, zero_point = fer.get_output_details()[0]['quantization']
        dequant = (raw.astype(np.float32) - zero_point) * scale
        norm = (dequant - dequant.min()) / (dequant.max() - dequant.min())
        emo = EMO[int(np.argmax(dequant))]
        ##print(f"scale = {scale}, zero_point = {zero_point}")
        ##print("Dequantized output:", dequant)
        print("Normalized:", norm)
        for i, v in enumerate(norm):
            print(f"[{i}] → {v:.2f}")
        data.append({'box':[x0,y0,x1,y1],'emo':emo})


    # ── 打印 per-frame det/fer FPS ──────────
    if fer_times:
        fer_fps = len(fer_times)/sum(fer_times)
        print(f"det.invoke() {det_fps:6.1f} FPS | fer.invoke() {fer_fps:6.1f} FPS")
        print(f"[INFO] num_faces = {len(fer_times)}")
        for i, t in enumerate(fer_times):
            print(f"    第 {i+1} 张脸推理耗时: {t*1000:.2f} ms")
    else:
        print(f"det.invoke() {det_fps:6.1f} FPS | fer.invoke() ----.- FPS (no face)")

    pub.send_json({'ts':time.time(),'data':data})
     # 每帧调用（非阻塞）
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_history.append(cpu_pct)
    
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024 / 1024
    mem_history.append(mem_MB)
    print(f"[INFO] 当前内存占用: {mem_MB:.1f} MB")
        
    print(f"[INFO] 当前 CPU 占用率: {psutil.cpu_percent(interval=0.1)}%")

    # ── 整管线 FPS 统计 ─────────────────────
    frame_cnt += 1                                      # <<< 新增
    if frame_cnt % FPS_WINDOW == 0:                     # <<< 新增
        overall_fps = FPS_WINDOW / (time.time() - t_start)
        print(f"⚡ Overall {overall_fps:5.1f} FPS  | last result →", data)
        ##print(f"[INFO] 当前 CPU 占用率: {psutil.cpu_percent(interval=0.1)}%")
        
        avg_cpu = sum(cpu_history[-FPS_WINDOW:]) / len(cpu_history[-FPS_WINDOW:])
        print(f"[INFO] 最近 {FPS_WINDOW} 帧平均 CPU 占用率: {avg_cpu:.1f}%")
  
        avg_mem = sum(mem_history[-FPS_WINDOW:]) / len(mem_history[-FPS_WINDOW:])
        print(f"[INFO] 最近 {FPS_WINDOW} 帧平均内存占用: {avg_mem:.1f} MB")

        t_start = time.time()                           # 重置窗口

