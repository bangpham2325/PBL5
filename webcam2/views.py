from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

sys.path.insert(0, './yolov5')
import argparse
import os
import platform
import shutil
from pathlib import Path
import cv2
import pandas as pd
import copy
import numpy as np
import time
# import dlib
import torch
from threading import Thread
import torch.backends.cudnn as cudnn
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from util.OPT_config import OPT
from util.common import read_yml, extract_xywh_hog
from datetime import datetime, timedelta
import pyrebase
from datetime import datetime
from dotenv import load_dotenv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []
sheet = {}
now = datetime.now()
env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path)


def upload_firebase():
    config = {
        "apiKey": os.getenv('apiKey'),
        "authDomain": os.getenv('authDomain'),
        "databaseURL": os.getenv('databaseURL'),
        "projectId": os.getenv('projectId'),
        "storageBucket": os.getenv('storageBucket'),
        "serviceAccount": os.getenv('serviceAccount')

    }
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    # db = firebase.database()
    if datetime.now().hour == 0 and datetime.now().minute == 0 and datetime.now().second == 0:
        storage.child(came_name + "_vehicle.csv.csv").put(came_name + "_vehicle.csv")


def count_obj(box, w, h, id):
    global count, data
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))
    if int(box[1] + (box[3] - box[1]) / 2) > (h - 350):
        if id not in data:
            count += 1
            data.append(id)


async def count_vehicle(request):
    return HttpResponse(count)


config = read_yml('settings/config.yml')

opt = OPT(config=config)

opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
df = pd.DataFrame(columns=["VehicleID", "Date", "Time", "Camera", "Speed", "Type"])
out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_csv, imgsz, evaluate, half = \
    opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
    opt.save_txt, opt.save_csv, opt.imgsz, opt.evaluate, opt.half
firebase = True

# initialize deepsort
cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

# Initialize
device = select_device(opt.device)
half &= device.type != 'cpu'  # half precision only suvehiclesorted on CUDA

if not evaluate:
    if os.path.exists(out):
        pass
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
device = select_device(device)
model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
stride, pt, jit, onnx = model.stride, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= pt and device.type != 'cpu'  # half precision only suvehiclesorted by PyTorch on CUDA
if pt:
    model.model.half() if half else model.model.float()

# Set Dataloader
vid_path, vid_writer = None, None
# Check if environment suvehiclesorts image displays

# if show_vid:
#     show_vid = check_imshow()


# Get names and colors
came_name = "cam2"
save_path = str(Path(out))
# extract what is in between the last '/' and last '.'
# csv_path = str(Path(out)) + '/' + came_name + '_vehicle' + '.csv'

if pt and device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup


def detect(df):
    source = "./videos/Traffic.mp4"
    filepath = Path('./data/cam2_vehicle.csv')
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size

    vid_path, vid_writer = [None] * bs, [None] * bs
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    names = ['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck']
    previous_frame, current_frame = [-1, -1]
    vehicle_infos = {}
    # id:{start in view, exit view, type }
    list_vehicles = set()  # LIST CONTAIN vehicles HAS APPEARED, IF THAT VEHICLE HAD BEEN UPLOADED TO DB, REMOVE THAT VEHICLE
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_path / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Avehiclesly NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # s += '%gx%g ' % img.shape[2:]  # print string
            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                # # pass detections to deepsort, only objects in used zone
                current_frame = {}
                current_frame['time'] = datetime.now()
                current_frame['frame'] = frame_idx
                current_frame['n_vehicles_at_time'] = len(outputs)
                current_frame['VehicleID'] = []

                if len(outputs) > 0:
                    current_frame['VehicleID'] = list(outputs[:, 4])
                    # current_frame['bb_vehicles'] = list(outputs[:, :4])

                if (current_frame != -1) and (previous_frame != -1):
                    previous_IDs = previous_frame['VehicleID']
                    current_IDs = current_frame['VehicleID']

                    for ID in current_IDs:
                        # neu id khong co trong khung hinh truoc va chua tung xuat hien
                        if (ID not in previous_IDs) and (ID not in list_vehicles):
                            try:
                                vehicle_infos[ID] = {}
                                vehicle_infos[ID]['in_time'] = datetime.now()
                                vehicle_infos[ID]['exit_time'] = datetime.max
                                vehicle_infos[ID]['Type'] = 'vehicle'
                                vehicle_infos[ID]['temporarily_disappear'] = 0
                            except:
                                pass


                    # for ID in previous_IDs:
                    for ID in copy.deepcopy(list_vehicles):
                        if ID not in current_IDs:
                            try:
                                vehicle_infos[ID]['Time'] = datetime.now()
                                vehicle_infos[ID]['temporarily_disappear'] += 1
                                if (vehicle_infos[ID]['temporarily_disappear'] > 75) and \
                                        (vehicle_infos[ID]['exit_time'] - vehicle_infos[ID]['in_time']) > timedelta(
                                    seconds=3):

                                    str_ID = str(ID)
                                    if opt.save_csv:
                                        df3 = pd.DataFrame([[str_ID, vehicle_infos[ID]['in_time'].strftime('%m/%d/%Y'),
                                                             vehicle_infos[ID]['in_time'].strftime('%H:%M'), came_name,
                                                             0,
                                                             vehicle_infos[ID]['Type']]],
                                                           columns=["VehicleID", "Date", "Time", "Camera", "Speed",
                                                                    "Type"])

                                        df = df.append(df3, ignore_index=True)
                                        Thread(target=df.to_csv("cam2_vehicle.csv", index=False),
                                               args=[]).start()
                                    list_vehicles.discard(ID)
                            except:
                                pass
                            # 25 frame ~ 1 seconds


                # Visualize deepsort outputs
                if len(outputs) > 0:

                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        # label = f'{id} {names[c]} {conf:.2f}'
                        count_obj(bboxes, w, h, id)
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        try:
                            vehicle_infos[id]['Type'] = names[c]
                        except:
                            pass

                    vehicles_count, IDs_vehicles = current_frame['n_vehicles_at_time'], current_frame[
                        'VehicleID']
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                    if not np.isnan(np.sum(IDs_vehicles)):
                        list_vehicles.update(list(IDs_vehicles))

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
            if show_vid:
                global count
                color = (0, 255, 0)
                start_point = (0, h - 350)
                end_point = (w, h - 350)
                cv2.line(im0, start_point, end_point, color, thickness=2)
                thickness = 3
                org = (150, 150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 3
                cv2.putText(im0, str(count), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow(str(p), im0)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            if firebase:
                Thread(target=upload_firebase, args=[]).start()

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        previous_frame = current_frame

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_vid or save_csv:
        print('Results saved to %s' % os.getcwd() + os.sep + out)


async def video_cam2(request):
    return StreamingHttpResponse(detect(df), content_type='multipart/x-mixed-replace; boundary=frame')
