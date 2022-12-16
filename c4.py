#!/usr/bin/env python3
import sys, os, time, threading, signal, math, subprocess, itertools
from functools import reduce
from unittest.mock import Mock
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
    kipr = Mock()

try:
    import IPython
    _ = IPython
except:
    print("no ipython")
    IPython = Mock()

old_print = print
start_time = time.time()
print = lambda *args, **kwargs: old_print(f"[{round(time.time() - start_time, 2)}]", *args, **kwargs)

def system(cmd):
    with os.popen(cmd) as subproc:
        return subproc.read().strip()

signal.signal(signal.SIGINT, lambda *_:  sys.exit(1))

# Keep the wombat screen on
def screen_on_loop():
    while True:
        system("xset dpms force on")
        system("xset s off")
        time.sleep(30)

RPI = system("uname -m") == "aarch64"
if RPI: threading.Thread(target=screen_on_loop, daemon=True).start()

HIT = False
def start_hit():
    global HIT
    HIT = False
    def await_hit(*args, **kwargs):
        global HIT
        input("Press enter to stop ")
        HIT = True
    threading.Thread(target=await_hit).start()

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

DEBUG = "sliders" in sys.argv
CHEATS = "cheats" in sys.argv

if DEBUG:
    SLIDERS = BLUE_SLIDERS
    SLIDERS += YELLOW_SLIDERS
    SLIDERS += RED_SLIDERS
    SLIDERS += BLUR_SLIDERS
else:
    SLIDERS = []

# Save current colors to colors.py
def save_colors(*_args, **_kwargs):
    if not DEBUG: return
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

WINDOW_MADE = False
def make_window():
    global WINDOW_MADE
    if WINDOW_MADE: return
    WINDOW_MADE = True
    cv2.namedWindow('Connect4', cv2.WINDOW_NORMAL)
    cv2.createButton("Save Colors", save_colors)
    for (name, nmax, arr, idx) in SLIDERS:
        def assign(arr, idx, val): arr[idx] = val
        if "BLUR" in name:
            lam = lambda n, arr=arr, idx=idx: assign(arr, idx, 2 * int(n / 2) + 1)
        else:
            lam = lambda n, arr=arr, idx=idx: assign(arr, idx, n)
        cv2.createTrackbar(name, 'Connect4', arr[idx], nmax, lam)

if __name__ == "__main__":
    make_window()

def make_get_mask(lower_upper_pairs):
    def get_mask(lups, frame):
        masks = map(lambda lup, frame=frame: cv2.inRange(frame, lup[0], lup[1]), lups)
        mask = reduce(lambda a, b: a + b, masks)
        # mask = cv2.inRange(frame, lower, upper)
        mask = cv2.GaussianBlur(mask, MASK_BLUR, 0)
        # mask = cv2.blur(mask, MASK_BLUR)
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
    make_window()
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
BLOB_RAD = 20
BLOB_RAD_ALLOWANCE = 10
def detect_blobs_in_contour(frame, contour):
    keypoints = BLOB_DET.detect(frame)
    keypoints = filter(lambda kp: abs(kp.size - BLOB_RAD) <= BLOB_RAD_ALLOWANCE, keypoints)
    keypoints = filter(lambda kp: cv2.pointPolygonTest(contour, kp.pt, False) > 0, keypoints)
    return list(keypoints)

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

def draw_text(frame, text, pos, thickness=1, fontScale=0.5, color=(255, 255, 255)):
    return cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(math.sqrt(dx * dx + dy * dy))

# This class assumes red always goes first!
# 'c' means color (not column! that's 'x')
# x, y are 0-indexed, except for in the c4solver imitation constructor
class Board:
    def __init__(self, board):
        # 1-indexed, like c4solver
        if type(board) == int:
            self.board = "_" * 42
            moves = list(map(lambda c: int(c) - 1, str(board)))
            red = True
            for x in moves:
                y = self.row_if_col_played(x)
                assert y != None, f"Invalid board, col {x} is full"
                i = Board.index(x, y)
                self.board = self.board[:i] + ("r" if red else "y") + self.board[i+1:]
                red = not red

        elif type(board) == str:
            assert len(board) == 42, f"{len(board)=}"
            self.board = board

        else:
            raise TypeError("board must be a str or int")
        self.assert_makes_sense()
        print(f"Board({self.board})")

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

    def row_if_col_played(self, x):
        for y in range(6):
            if self.piece_at(x, y) == "_":
                return y
        return None

    def whos_turn_is_it(self):
        return 'r' if self.board.count("_") % 2 == 0 else 'y'

    def spaces_empty(self):
        return self.board.count("_")

    def assert_makes_sense(self):
        for x in range(7):
            for y in range(1,6):
                if self.piece_at(x, y) != "_":
                    assert self.piece_at(x, y-1) != "_", "Floating piece"
                else:
                    break

    def n_connected(self, c, x, y, dx, dy):
        n = 0
        while self.piece_at_oob_safe(x + dx, y + dy) == c:
            n += 1
            x += dx
            y += dy
        return n

    def is_winning_move(self, c, x):
        y = self.row_if_col_played(x)
        nc = lambda dx, dy: self.n_connected(c, x, y, dx, dy)
        lr = nc(1, 0) + nc(-1, 0)
        ud = nc(0, 1) + nc(0, -1)
        uldr = nc(-1, 1) + nc(1, -1)
        dlur = nc(-1, -1) + nc(1, 1)
        return lr >= 3 or ud >= 3 or uldr >= 3 or dlur >= 3

    def get_next_move(self):
        if os.path.isfile("/home/user/connect4/c4solver"):
            path = "/home/user/connect4/c4solver"
        else:
            path = "/connect4/c4solver"

        moves, red, yellow, mask = self.format_c4solver()
        cur_player = red if moves % 2 == 0 else yellow
        line = ("#" + str(cur_player) + " " + str(mask) + " " + str(moves) + "\n").encode('utf-8')

        print(f"{line=}")
        p = subprocess.Popen([path], cwd=os.path.dirname(path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.stdin.write(line)
        p.stdin.close()
        answer = p.stdout.readline().decode('utf-8').split()
        p.wait(5)

        print(answer)

        max_i = list(sorted(enumerate(map(int, answer)), key=snd))[-1][0]
        print(f"I want to play column {max_i}")
        return max_i

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
                mask_n |= 1 << i
                moves += 1
            elif c == 'y':
                yellow_n |= 1 << i
                mask_n |= 1 << i
                moves += 1

        return moves, red_n, yellow_n, mask_n

# Motor 1 = wheel
# Motor 2 = magazine
# Servo 0 = grabber
# Analog 0 = magazine dist
# Digital 0 = refill bump button
class Robot:
    def __init__(self):
        self.pos = 0
        kipr.enable_servo(0)
        self.unclamp()

    def clamp(self):
        kipr.set_servo_position(0, 500)
        time.sleep(0.3)

    def unclamp(self):
        kipr.set_servo_position(0, 800)
        time.sleep(0.3)

    def move(self, amt):
        self.pos += amt
        sign = 1 if amt > 0 else -1
        speed = sign * 400
        kipr.mav(0, speed)
        time.sleep(sign * amt)
        kipr.ao()

    def move_to_pos(self, pos):
        if pos != self.pos:
            self.move(pos - self.pos)

    def move_to(self, column):
        POSITIONS = { # 0 is all the way left, so it's the furthest
            0: 4.80,
            1: 4.25,
            2: 3.65,
            3: 2.9,
            4: 2.1,
            5: 1.35,
        }
        if column == 'refill':
            kipr.mav(0, -300)
            while kipr.digital(0) == 0:
                time.sleep(0.025)
            kipr.ao()
            self.pos = 0
            return

        if CHEATS:
            start_hit()
            kipr.mav(0, 300)
            while not HIT:
                time.sleep(0.05)
            kipr.ao()
            return

        assert column in POSITIONS, f"unknown position {column}"
        self.move_to_pos(POSITIONS[column])

    def mag_wind(self):
        WIND_SPEED = 400
        kipr.mav(1, WIND_SPEED)
        while kipr.analog(0) > 2500:
            time.sleep(5 / 1000)
        kipr.ao()

    def mag_unwind(self):
        kipr.mav(1, -800)
        time.sleep(5)
        kipr.ao()

def wait_key():
    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):
        sys.exit()
    elif key == ord('s'):
        save_colors()
    elif key == 13: # enter
        pass
    elif key == 43: # plus
        pass
    elif key == 45: # minus
        pass
    elif key == 8: # bksp
        pass


VID = None
def get_frame():
    global VID
    if VID == None:
        VID = open_webcam()
    _ret, frame = VID.read()
    return frame

PIECE_NEAR_POINT_RADIUS = 5/320 * CAP_RESOLUTION[0]
def fst(l): return l[0]
def snd(l): return l[1]
def divine_board(ps, ys, rs):
    # Verify a sensible board
    assert len(ps) == 42, "not 42 keypoints"
    assert len(ys) <= 20, "more than 20 yellow pieces"
    assert len(rs) <= 21, "more than 21 red pieces"

    # Extract raw point from wrapper object
    mappt = lambda i: list(map(lambda p: p.pt, i))
    ps = mappt(ps)
    ys = mappt(ys)
    rs = mappt(rs)

    # Sort pieces, left-to-right, bottom-to-top
    ps_l2r = list(sorted(ps, key=fst))
    lrss = lambda l: list(reversed(sorted(l, key=snd)))
    ps_l2r_b2t = lrss(ps_l2r[:6]) + lrss(ps_l2r[6:12]) \
                  + lrss(ps_l2r[12:18]) + lrss(ps_l2r[18:24]) \
                  + lrss(ps_l2r[24:30]) + lrss(ps_l2r[30:36]) \
                  + lrss(ps_l2r[36:42])

    ps_sorted = list(map(lambda p: (int(p[0]), int(p[1])), ps_l2r_b2t))

    # Create a board string from sorted keypoint positions
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

def read_board_one_frame():
        # Get frame
        frame = get_frame()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color-based masks
        board_mask = get_board_mask(hsv)
        yellow_mask = get_yellow_piece_mask(hsv)
        red_mask = get_red_piece_mask(hsv)

        # Get the contour around the board
        board_contours, _hierarchy = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(board_contours) == 0:
            return { "error": "No board contour", "frame": frame, "final": frame, "board_mask": board_mask, "yellow_mask": yellow_mask, "red_mask": red_mask }
        board_contour = max(board_contours, key=cv2.contourArea)
        board_only_mask = np.zeros(board_mask.shape, board_mask.dtype)
        cv2.fillPoly(board_only_mask, pts=[board_contour], color=255)

        # Detect piece positions
        pos_kps = detect_blobs_in_contour(board_mask, board_contour)
        y_kps = detect_blobs_in_contour(cv2.bitwise_not(yellow_mask), board_contour)
        r_kps = detect_blobs_in_contour(cv2.bitwise_not(red_mask), board_contour)

        # Divine the board (read the board from keypoint information)
        divined = None
        divined_err = None
        try:
            if len(pos_kps) == 42 and len(y_kps) <= 20 and len(r_kps) <= 21:
                divined = divine_board(pos_kps, y_kps, r_kps)
            else:
                divined_err = f"divining: {len(pos_kps)=} {len(y_kps)=} {len(r_kps)=}"
        except Exception as e:
            divined_err = f"divining: {e}"

        # Make drawing frames
        final = apply_mask(frame, board_only_mask)

        # Draw stuff on drawing frames
        frame = draw_contour(frame, board_contour, color=(255, 0, 0), width=4)
        board_mask = draw_keypoints(grayToBGR(board_mask), pos_kps, (255, 0, 255))
        final = draw_keypoints(final, y_kps, (0, 255, 255))
        final = draw_keypoints(final, r_kps, (0, 0, 255))

        if divined_err:
            return { "error": divined_err, "frame": frame, "final": final, "board_mask": board_mask, "yellow_mask": yellow_mask, "red_mask": red_mask }

        if divined:
            sorted_pos_kps = divined[0]
            board = divined[1]
            for (i, p) in enumerate(sorted_pos_kps):
                offset = 8
                p = (p[0] - offset, p[1] + offset)
                if board[i] == "_":
                    color = (255, 255, 255)
                elif board[i] == "r":
                    color = (0, 0, 255)
                else: # board[i] == "y":
                    color = (0, 255, 255)
                final = draw_text(final, str(i), p, color=color)

        return {
            "frame": frame,
            "board_mask": board_mask,
            "yellow_mask": yellow_mask,
            "red_mask": red_mask,
            "final": final,
            "divined": divined,
            "error": None,
        }


READ_BOARDS = 20 # Number of frames where we can parse a board, before we finish "reading"
ACCEPTABLE_CONFIDENCE = 0.85 # Fraction of identical frames to be acceptable
def read_board():
    while True:
        boards = []
        while len(boards) < READ_BOARDS:
            read = read_board_one_frame()
            if read["error"] != None:
                print("read_board:", read["error"])
                frame = read["frame"]
                final = read.get("final")
                if type(final) != type(None): frame = final
                frame = draw_text(frame, f"{len(boards)}", (0, 50), color=(0, 0, 255), fontScale=2, thickness=2)
                imshow_n([frame])
                wait_key()
                continue

            board_str = read["divined"][1]
            boards.append(board_str)
            final = read["final"]
            final = draw_text(final, str(len(boards)), (0, 50), fontScale=2, thickness=2)

            imshow_n([final])
            wait_key()

        # Get most common board, ensure we're confident enough in it.
        preferred_board = max(boards, key=boards.count)
        n_preferred = boards.count(preferred_board)
        if n_preferred / len(boards) > ACCEPTABLE_CONFIDENCE:
            return preferred_board, read["frame"], read["divined"][0]

def show_message(msg):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img = draw_text(img, msg, (10, 200), thickness=2, fontScale=2, color=(255, 255, 255))
    imshow_n([img])
    wait_key()

def play():
    r = Robot()
    first = True
    show_message("My Turn!")
    time.sleep(2)
    board_str, frame, pos_kps = read_board()
    while True:
        board = Board(board_str)

        if first:
            first = False
        else:
            show_message("My Turn!")
            time.sleep(2)

        if board.spaces_empty() <= 1:
            show_message("Draw!")
            time.sleep(5)
            return

        x = board.get_next_move()
        y = board.row_if_col_played(x)
        my_color = board.whos_turn_is_it()
        will_win = board.is_winning_move(my_color, x)

        bottom_kp = pos_kps[x*6]
        top_kp = pos_kps[x*6 + 5]
        cv2.line(frame, bottom_kp, top_kp, (0, 255, 0), 3)
        kp_to_play = pos_kps[x*6 + y]
        cv2.circle(frame, kp_to_play, 5, (0, 0, 255) if my_color == 'r' else (0, 255, 255), -1)
        imshow_n([frame])
        wait_key()
        time.sleep(2)

        input("Press enter when piece ready. ")
        r.clamp()
        time.sleep(0.5)
        r.move_to(x)
        time.sleep(0.5)
        r.unclamp()
        time.sleep(0.5)
        r.move_to("refill")
        time.sleep(0.5)

        if will_win:
            won_board_str = read_board()[0]
            won_board = Board(won_board_str)
            if won_board.piece_at(x, y) == my_color:
                show_message("I Won!")
                time.sleep(5)
                return

        while True:
            show_message("Your Turn!")
            time.sleep(5)
            new_board_str, new_frame, new_pos_kps = read_board()
            new_board = Board(new_board_str)
            if board.spaces_empty() - new_board.spaces_empty()  == 2:
                board_str = new_board_str
                frame = new_frame
                pos_kps = new_pos_kps
                break


def watch_mask():
    while True:
        x = read_board_one_frame()
        if x["error"] != None:
            print(x["error"])
        frames = [x["frame"], x["board_mask"], x["yellow_mask"], x["red_mask"], x["final"]]
        imshow_n(frames)
        wait_key()

if __name__ == "__main__":
    if DEBUG:
        watch_mask()
    else:
        play()
