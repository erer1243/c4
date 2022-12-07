#!/usr/bin/env python3
import sys, os, time, threading, signal, math, subprocess, pprint, itertools
from functools import reduce
import numpy as np
import cv2

sys.dont_write_bytecode = True
import colors
from colors import (
    LOWER_BLUE, UPPER_BLUE, LOWER_YELLOW, UPPER_YELLOW,
    UPPER_RED_1, LOWER_RED_1, UPPER_RED_2, LOWER_RED_2, MASK_BLUR
)

try:
    import kipr
    _ = kipr
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

signal.signal(signal.SIGINT, lambda *_:  sys.exit(1))

def screen_on_loop():
    while True:
        system("xset dpms force on")
        time.sleep(180)

RPI = system("uname -m") == "aarch64"
if RPI: threading.Thread(target=screen_on_loop, daemon=True).start()

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
SLIDERS += BLUE_SLIDERS
SLIDERS += YELLOW_SLIDERS
# SLIDERS += RED_SLIDERS
# SLIDERS += BLUR_SLIDERS
if RPI: SLIDERS = []

def save_colors(*_args, **_kwargs):
    print("Saving colors")
    members = filter(lambda s: s.isupper(), dir(colors))
    with open(colors.__file__, 'w') as f:
        f.write("import numpy as np\n")
        for name in members:
            try:
                new_val = str(list(globals()[name]))
                line = f"{name} = np.array({new_val})\n"
                f.write(line)
            except:
                pass

if __name__ == "__main__":
    cv2.namedWindow('Connect4', cv2.WINDOW_NORMAL)
    cv2.createButton("Save Colors", save_colors)
    for (name, nmax, arr, idx) in SLIDERS:
        def assign(arr, idx, val): arr[idx] = val
        if "BLUR" in name:
            lam = lambda n, arr=arr, idx=idx: assign(arr, idx, 2 * int(n / 2) + 1)
        else:
            lam = lambda n, arr=arr, idx=idx: assign(arr, idx, n)
        cv2.createTrackbar(name, 'Connect4', arr[idx], nmax, lam)

def make_get_mask(lower_upper_pairs):
    def get_mask(lups, frame):
        masks = map(lambda lup, frame=frame: cv2.inRange(frame, lup[0], lup[1]), lups)
        mask = reduce(lambda a, b: a + b, masks)
        # mask = cv2.inRange(frame, lower, upper)
        mask = cv2.GaussianBlur(mask, MASK_BLUR, 0)
        return mask
    return lambda frame, lups=lower_upper_pairs: get_mask(lups, frame)

get_board_mask = make_get_mask([(LOWER_BLUE, UPPER_BLUE)])
get_yellow_piece_mask = make_get_mask([(LOWER_YELLOW, UPPER_YELLOW)])
get_red_piece_mask = make_get_mask([(LOWER_RED_1, UPPER_RED_1), (LOWER_RED_2, UPPER_RED_2)])

def imshow_n(frames):
    # Convert masks to BGR
    for i in range(0, len(frames)):
        f = frames[i]
        if len(f.shape) == 2:
            frames[i] = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

    catted = np.concatenate(frames, axis=1)
    cv2.imshow('Connect4', catted)

def grayToBGR(*frames):
    t = tuple(map(lambda f: cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), frames))
    if len(t) == 1:
        return t[0]
    else:
        return t

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

def draw_contour(frame, contour, **kwargs):
    return draw_contours(frame, [contour], **kwargs)

def draw_hull(frame, hull, color=(0,255,0), width=2):
    copy = np.array(frame)
    cv2.drawContours(copy, [hull], -1, color, width)
    return copy

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(math.sqrt(dx * dx + dy * dy))

PIECE_NEAR_POINT_RADIUS = 5/320 * CAP_RESOLUTION[0]
def fst(l): return l[0]
def snd(l): return l[1]
def divine_board(ps, ys, rs):
    # Verify a sensible board
    assert len(ps) == 42, "not 42 keypoints"
    assert len(ys) <= 20, "more than 20 yellow pieces"
    assert len(rs) <= 21, "more than 21 red pieces"

    mappt = lambda i: list(map(lambda p: p.pt, i))
    ps = mappt(ps)
    ys = mappt(ys)
    rs = mappt(rs)

    ps_l2r = list(sorted(ps, key=fst))
    lrss = lambda l: list(reversed(sorted(l, key=snd)))
    ps_l2r_b2t = lrss(ps_l2r[:6]) + lrss(ps_l2r[6:12]) \
                  + lrss(ps_l2r[12:18]) + lrss(ps_l2r[18:24]) \
                  + lrss(ps_l2r[24:30]) + lrss(ps_l2r[30:36]) \
                  + lrss(ps_l2r[36:42])

    ps_sorted = list(map(lambda p: (int(p[0]), int(p[1])), ps_l2r_b2t))

    board = "_" * 42
    pieces_in_board = 0
    for (c, p) in itertools.chain(map(lambda p: ('y', p), ys), map(lambda p: ('r', p), rs)):

        near_ps = list(sorted(map(lambda ip, p_=p: (ip[0], dist(ip[1], p_)), enumerate(ps_sorted)), key=snd))
        near_p = near_ps[0]
        if near_p[1] <= PIECE_NEAR_POINT_RADIUS:
            i = near_p[0]
            assert board[i] == "_", "two pieces in the same spot?"
            board = board[:i] + c + board[i+1:]
            pieces_in_board += 1

    assert pieces_in_board == len(ys) + len(rs), "some pieces are in a weird spot"
    return ps_sorted, board

def strrep(s, i, c):
    return s[:i] + c + s[i+1:]

# This class assumes red always goes first!
class Board:
    def __init__(self, board):
        # 1-indexed, like c4solver
        if type(board) == int:
            self.board = "_" * 42
            red = True
            while board > 0:
                x = (board % 10) - 1
                board = int(board / 10)
                for y in range(6):
                    if self.piece_at(x, y) == "_":
                        i = Board.index(x, y)
                        self.board = strrep(self.board, i, "r" if red else "y")
                        break
                red = not red
        elif type(board) == str:
            assert len(board) == 42, f"{len(board)=}"
            self.board = board
        else:
            raise TypeError("board must be a str or int")

        self.assert_makes_sense()

    def index(x, y):
        assert y < 6 and y >= 0, "y out of bounds"
        assert x < 7 and x >= 0, "x out of bounds"
        return x * 6 + y

    def piece_at(self, x, y):
        i = Board.index(x, y)
        return self.board[i]

    def piece_at_oob_safe(self, x, y):
        if x < 0 or x >= 7 or y < 0 or y >= 6:
            return '_'
        else:
            return self.piece_at(x, y)

    def get_next_move(self):
        if os.path.isfile("/home/user/connect4/c4solver"):
            path = "/home/user/connect4/c4solver"
        else:
            path = "/connect4/c4solver"

        moves, red, yellow, mask = self.format_c4solver()
        cur_player = red if moves % 2 == 0 else yellow
        line = ("#" + str(cur_player) + " " + str(mask) + " " + str(moves) + "\n").encode('utf-8')

        p = subprocess.Popen([path], cwd=os.path.dirname(path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.stdin.write(line)
        p.stdin.close()
        answer = p.stdout.readline().decode('utf-8').split()
        p.wait(5)

        max_i = next(sorted(enumerate(answer), key=snd))[0]
        return max_i

    def assert_makes_sense(self):
        for x in range(7):
            for y in range(1,6):
                if self.piece_at(x, y) != "_":
                    assert self.piece_at(x, y-1) != "_", "Floating piece"
                else:
                    break

    def format_c4solver(self):
        # BOTTOM_N = 1 | 1 << 7 | 1 << 14 | 1 << 21 | 1 << 28 | 1 << 35 | 1 << 42
        board_spaced = ""
        for x in range(7):
            for y in range(6):
                board_spaced += self.piece_at(x, y)
            board_spaced += "_"

        red_n = 0
        yellow_n = 0
        mask_n = 0
        moves = 0
        for (i, c) in enumerate(board_spaced):
            if c == 'r':
                red_n |= 1 << i
                moves += 1
            elif c == 'y':
                yellow_n |= 1 << i
                moves += 1

        return moves, red_n, yellow_n, mask_n

# BOARD = "_" * 42

# def get_board_at(x, y):
#     global BOARD
#     if x < 0 or x >= 7 or y < 0 or y >= 6:
#         return '_'
#     i = pos_to_idx(x, y)
#     return BOARD[i]

# def set_board_at(c, x, y):
#     global BOARD
#     i = pos_to_idx(x, y)
#     return BOARD[:i] + c + BOARD[i+1:]

#     # print(format(pos_n, 'b'), format(mask_n, 'b'))
#     # print(f"{pos_n=} {mask_n=} {pos_n + mask_n=} {pos_n + mask_n + BOTTOM_N=}")
#     print(f"{pos_n=} {mask_n=} {moves=}")
#     return pos_n, mask_n, moves

# def get_next_move():

# def play_pos(c, x, y):
#     global BOARD
#     old_board = BOARD
#     try:
#         pos_to_idx(x, y) # assert good indices
#         assert c == 'r' or c == 'y', "invalid char"
#         assert get_board_at(x, y) == '_', "spot already taken"
#         new_board = set_board_at(c, x, y)
#         assert len(BOARD) == len(new_board), "invalid new len"
#         assert y == 0 or get_board_at(x, y-1) != "_", "floating piece"
#         BOARD = new_board
#         assert board_makes_sense(), "board senseless"
#         return True
#     except Exception as e:
#         BOARD = old_board
#         return False, str(e)


READ_SECS = 4
def read_board():
    start = time.time()
    while time.time() - start < READ_SECS:
        pass

def wait_key():
    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):
        sys.exit()
    elif key == ord('s'):
        save_colors()
    elif key == 13: # enter
        print(BOARD)
        if board_makes_sense():
            get_next_move()
        else:
            print("Board is senseless!")
    elif key == 43: # plus
        pass
    elif key == 45: # minus
        pass
    elif key == 8: # bksp
        pass

if __name__ == "__main__":
    vid = open_webcam()

def read_board_one_frame():
        # Get frame
        _ret, frame = vid.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color-based masks
        board_mask = get_board_mask(hsv)
        yellow_mask = get_yellow_piece_mask(hsv)
        red_mask = get_red_piece_mask(hsv)

        # Get the contour around the board
        board_contours, _hierarchy = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(board_contours) == 0:
            return "No board contour", frame
        board_contour = max(board_contours, key=cv2.contourArea)

        # Detect piece positions
        pos_kps = detect_blobs_in_contour(board_mask, board_contour)
        y_kps = detect_blobs_in_contour(cv2.bitwise_not(yellow_mask), board_contour)
        r_kps = detect_blobs_in_contour(cv2.bitwise_not(red_mask), board_contour)

        divined = None
        try:
            if len(pos_kps) == 42 and len(y_kps) <= 20 and len(r_kps) <= 21:
                divined = divine_board(pos_kps, y_kps, r_kps)
        except Exception as e:
            print(e)

        # Make drawing frames
        final = apply_mask(frame, board_mask)

        # Draw stuff on drawing frames
        frame = draw_contour(frame, board_contour, color=(255, 0, 0), width=4)
        board_mask = draw_keypoints(grayToBGR(board_mask), pos_kps, (255, 0, 255))
        # final = draw_keypoints(final, pos_kps, (255, 255, 255))
        final = draw_keypoints(final, y_kps, (0, 255, 255))
        final = draw_keypoints(final, r_kps, (0, 0, 255))

        if divined:
            sorted_pos_kps = divined[0]
            board = divined[1]
            global BOARD
            BOARD = board
            for (i, p) in enumerate(sorted_pos_kps):
                offset = 8
                p = (p[0] - offset, p[1] + offset)
                if board[i] == "_":
                    color = (255, 255, 255)
                elif board[i] == "r":
                    color = (0, 0, 255)
                else: # board[i] == "y":
                    color = (0, 255, 255)
                final = cv2.putText(final, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return {
            "frame": frame,
            "board_mask": board_mask,
            "yellow_mask": yellow_mask,
            "red_mask": red_mask,
            "final": final
        }

def watch_mask():
    while True:

        x = read_board_one_frame()

        if type(x) == tuple: # Error
            frames = [x[1]]
        else: # No error
            frames = [x["board_mask"], x["yellow_mask"], x["red_mask"], x["final"]]
            # frames = list(x.values())

        imshow_n(frames)

        wait_key()

if __name__ == "__main__":
    watch_mask()
