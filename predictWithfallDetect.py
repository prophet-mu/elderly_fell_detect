# -*- coding: utf-8 -*-

import time
from ax import pipeline
import numpy as np
from PIL import Image, ImageDraw
import socket

IP="192.168.43.91"

pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/root/elderly_fell_detect/hrnet_pose_yolov8.json',
    '-c', '2',
])


def send_msg(msg):
    print("---------------------------------udp------------------------------")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host = IP
    port = 8888
    message = msg
    udp_socket.sendto(message.encode(), (host, port))
    print('send:', message)
    udp_socket.close()

lcd_width, lcd_height = 854, 480

img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
ui = ImageDraw.ImageDraw(img)

def rgba2argb(rgba):
    r,g,b,a = rgba.split()
    return Image.merge("RGBA", (a,b,g,r))
canvas_argb = rgba2argb(img)


WINDOW_SIZE = 5
THRESHOLD_ANGLE=30
isFall = False
detection = []
tag = 0
while pipeline.work():
    #time.sleep(0.001)
    argb = canvas_argb.copy()
    tmp = pipeline.result()
    if tmp and tmp['nObjSize']:
        ui = ImageDraw.ImageDraw(argb)
        for i in tmp['mObjects']:
            x = i['bbox']['x'] * lcd_width
            y = i['bbox']['y'] * lcd_height
            w = i['bbox']['w'] * lcd_width
            h = i['bbox']['h'] * lcd_height
            objprob = str(isFall)
            ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255))
            ui.text((x,y+20), str(objprob))
            if tag <5:
                detection.append(i)
                tag+=1
            else:
               # print("begin")
                head_filtered = np.zeros(2)
                foot_filtered = np.zeros(2)
                for j in range(len(detection)):
                    landmark = detection[j]['landmark']
                    head = np.array([landmark[0]['x'], landmark[0]['y']])
                    foot = np.array([landmark[10]['x'], landmark[10]['y']])
                    head_filtered += head / WINDOW_SIZE
                    foot_filtered += foot / WINDOW_SIZE
                vec = head_filtered - foot_filtered
                angle = np.arctan2(vec[1], vec[0]) * 180.0 / np.pi
                print(angle)
                if abs(angle) > 150 or abs(angle) <30:
                    print('The person has fallen down!')
                    isFall=True
                    send_msg("detect fall")
                    time.sleep(5)
                    print(angle)
                else:
                    print('The person is standing up!')
                    isFall=False
                detection = []
                tag = 0

            
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
    pipeline.config("display", (lcd_width, lcd_height, "ARGB", argb.tobytes()))
pipeline.free()
            
