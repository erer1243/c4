#!/usr/bin/env python3
import sys, os, time, threading, signal, random, math, logging
import numpy as np
import cv2
import serial

try:
    import kipr
except:
    print("no kipr")

try:
    import IPython
    debug = IPython.embed
except:
    print("no ipython")
    debug = lambda: None

def system(cmd):
    with os.popen(cmd) as subproc:
        return subproc.read().strip()

signal.signal(signal.SIGINT, lambda _a, _b:  sys.exit(1))

def screen_on_loop():
    while True:
        system("xset dpms force on")
        time.sleep(180)

RPI = system("uname -m") == "aarch64"
if RPI: threading.Thread(target=screen_on_loop, daemon=True).start()

def open_arduino_serial():
    return serial.Serial('/dev/ttyACM0', baudrate=115200)

LOWER_BLUE = np.array([105, 60, 60])
UPPER_BLUE = np.array([135, 255, 255])
LOWER_YELLOW = np.array([20, 150, 60])
UPPER_YELLOW = np.array([30, 255, 255])
LOWER_RED_1 = np.array([0, 175, 125])
UPPER_RED_1 = np.array([12, 255, 255])
LOWER_RED_2 = np.array([160, 125, 125])
UPPER_RED_2 = np.array([179, 255, 255])
MASK_BLUR = [3, 3]
YELLOW_SLIDERS = [
    ('YLH', 179, LOWER_YELLOW, 0),
    ('YLS', 255, LOWER_YELLOW, 1),
    ('YLV', 255, LOWER_YELLOW, 2),
    ('YUH', 179, UPPER_YELLOW, 0),
    ('YUS', 255, UPPER_YELLOW, 1),
    ('YUV', 255, UPPER_YELLOW, 2),
]
BLUE_SLIDERS = [
    ('BLH', 179, LOWER_BLUE, 0),
    ('BLS', 255, LOWER_BLUE, 1),
    ('BLV', 255, LOWER_BLUE, 2),
    ('BUH', 179, UPPER_BLUE, 0),
    ('BUS', 255, UPPER_BLUE, 1),
    ('BUV', 255, UPPER_BLUE, 2),
]
RED_SLIDERS = [
    ('R1LH', 179, LOWER_RED_1, 0),
    ('R1LS', 255, LOWER_RED_1, 1),
    ('R1LV', 255, LOWER_RED_1, 2),
    ('R1UH', 179, UPPER_RED_1, 0),
    ('R1US', 255, UPPER_RED_1, 1),
    ('R1UV', 255, UPPER_RED_1, 2),
    ('R2LH', 179, LOWER_RED_2, 0),
    ('R2LS', 255, LOWER_RED_2, 1),
    ('R2LV', 255, LOWER_RED_2, 2),
    ('R2UH', 179, UPPER_RED_2, 0),
    ('R2US', 255, UPPER_RED_2, 1),
    ('R2UV', 255, UPPER_RED_2, 2),
]
BLUR_SLIDERS = [
    ("BLURX", 100, MASK_BLUR, 0),
    ("BLURY", 100, MASK_BLUR, 1),
]
SLIDERS = []
# SLIDERS += BLUE_SLIDERS
# SLIDERS += YELLOW_SLIDERS
# SLIDERS += RED_SLIDERS
# SLIDERS += BLUR_SLIDERS
if RPI: SLIDERS = []

def make_get_mask(lower, upper):
    def get_mask(lower, upper, frame):
        mask = cv2.inRange(frame, lower, upper)
        mask = cv2.GaussianBlur(mask, MASK_BLUR, 0)
        return mask
    return lambda frame, lower=lower, upper=upper: get_mask(lower, upper, frame)

get_board_mask        = make_get_mask(LOWER_BLUE, UPPER_BLUE)
get_yellow_piece_mask = make_get_mask(LOWER_YELLOW, UPPER_YELLOW)

get_red_1 = make_get_mask(LOWER_RED_1, UPPER_RED_1)
get_red_2 = make_get_mask(LOWER_RED_2, UPPER_RED_2)
get_red_helper = lambda frame: get_red_1(frame) + get_red_2(frame)
get_red_piece_mask = lambda frame: cv2.GaussianBlur(get_red_helper(frame), MASK_BLUR, 0)

cv2.namedWindow('Connect4', cv2.WINDOW_NORMAL)
for (name, nmax, arr, idx) in SLIDERS:
    def assign(arr, idx, val): arr[idx] = val
    if "BLUR" in name:
        lam = lambda n, arr=arr, idx=idx: assign(arr, idx, 2 * int(n / 2) + 1)
    else:
        lam = lambda n, arr=arr, idx=idx: assign(arr, idx, n)
    cv2.createTrackbar(name, 'Connect4', arr[idx], nmax, lam)

def imshow_n(frames):
    for i in range(0, len(frames)):
        f = frames[i]
        if type(f) == tuple:
            frames[i] = cv2.cvtColor(f[0], f[1])
    catted = np.concatenate(frames, axis=1)
    cv2.imshow('Connect4', catted)

# CAP_RESOLUTION = (160, 120)
CAP_RESOLUTION = (320, 240)
# CAP_RESOLUTION = (424, 240)
# CAP_RESOLUTION = (640, 360)
# CAP_RESOLUTION = (1920, 1080)
def open_webcam():
    subproc = os.popen('v4l2-ctl --list-devices 2>/dev/null | grep -A1 Microsoft | tail -n1')
    with subproc as sp: device = sp.read().strip()
    if len(device) == 0:
        raise Exception("Can't find webcam")
    vid = cv2.VideoCapture(device)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_RESOLUTION[0])
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_RESOLUTION[1])
    return vid

def apply_mask(frame, mask): 
    return cv2.bitwise_and(frame, frame, mask=mask)

BLOB_DET = cv2.SimpleBlobDetector_create()
def detect_blobs_in_contour(frame, contour):
    keypoints = BLOB_DET.detect(frame)
    kps_in_contour = filter(lambda kp: cv2.pointPolygonTest(contour, kp.pt, False) > 0, keypoints)
    return list(kps_in_contour)

def draw_keypoints(frame, keypoints, color=(255, 255, 0)):
    return cv2.drawKeypoints(frame, keypoints, np.array([]), color)

def draw_contours(frame, contours, color=(0, 0, 255), width=2):
    copy = np.array(frame)
    cv2.drawContours(copy, contours, -1, color, width)
    return copy

def draw_hull(frame, hull, color=(0,255,0), width=2):
    copy = np.array(frame)
    cv2.drawContours(copy, [hull], -1, color, width)
    return copy
    
# def draw_line()

def dist(a, b):
    dx = a[0] - b[0] 
    dy = a[1] - b[1]
    return int(math.sqrt(dx * dx + dy * dy))

def fst(l): l[0]
def snd(l): l[1]
def divine_board(ps, ys, rs):
    # Verify a sensible board
    if len(ps) != 42:
        return "Not enough position markers"
    # for kp in ps:
    kp = ps[0]
    dist_to_kp = lambda kp2, kp=kp: [kp2, dist(kp.pt, kp2.pt)]
    ps_close = sorted(map(dist_to_kp, ps), key=snd)
    return list(map(fst, ps_close))
    
def watch_mask():
    vid = open_webcam()
    x = 0
    while True:
        _ret, frame = vid.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        board_mask = get_board_mask(hsv)
        yellow_mask = get_yellow_piece_mask(hsv)
        red_mask = get_red_piece_mask(hsv)

        board_contours, _hierarchy = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(board_contours) == 0: continue

        c_large = max(board_contours, key=cv2.contourArea)
        final = apply_mask(frame, board_mask)
        frame = draw_contours(frame, [c_large], color=(255, 0, 0), width=4)

        hull = cv2.convexHull(c_large)
        final = draw_hull(final, hull)

        pos_keypoints = detect_blobs_in_contour(board_mask, hull)
        y_keypoints = detect_blobs_in_contour(cv2.bitwise_not(yellow_mask), hull)
        r_keypoints = detect_blobs_in_contour(cv2.bitwise_not(red_mask), hull)
        final = draw_keypoints(final, pos_keypoints, (255, 255, 255))
        final = draw_keypoints(final, y_keypoints, (0, 255, 255))
        final = draw_keypoints(final, r_keypoints, (0, 0, 255))

        # x += 1
        # if x % 100 == 0:
        # x = divine_board(pos_keypoints, y_keypoints, r_keypoints)
        # if type(x) == list:
        #     print(x)

        imshow_n([
            frame, 
            (board_mask, cv2.COLOR_GRAY2BGR), 
            (yellow_mask, cv2.COLOR_GRAY2BGR), 
            (red_mask, cv2.COLOR_GRAY2BGR), 
            final,
        ])

        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break

watch_mask()
