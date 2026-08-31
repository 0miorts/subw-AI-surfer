from collections import deque
import cv2
import pyautogui
import mss
import mss.tools
import torch
import numpy as np
import time
import easyocr


class Environment:
    def __init__(self):
        self.coins = 0
        self.frames = deque(maxlen=4)
        self.score_region = {"top": 27, "left": 1740, "width": 165 , "height": 50}
        self.pause_region = {"top": 35, "left": 25, "width": 65 , "height": 50}
        self.gameplay_region = {"top": 240, "left": 350, "width": 1220 , "height": 840}
        self.score = 0
        self.stall_count = 0
        self.stall_threshold = 2
        self.game_state = "MENU"
        self.pause_button = cv2.imread("imgs/pause.png", cv2.IMREAD_GRAYSCALE)
        self.play_button_template = cv2.imread("imgs/play.png", cv2.IMREAD_GRAYSCALE)
        self.play_text = cv2.imread("imgs/first_play.png", cv2.IMREAD_GRAYSCALE)
        self.reader = easyocr.Reader(['en'], gpu=True)

    def reset(self):
        self.coins = 0
        self.score = None
        self.stall_count = 0
        self.frames.clear()
        # ------SKIP MENU----------
        while not (self.is_play_button() or self.is_start_text()):
            pyautogui.press('enter')
            time.sleep(0.2)
        if (self.is_play_button() or self.is_start_text()):
            pyautogui.press('enter')
            time.sleep(2.2)
            self.game_state = "GAME"
            print('----STARTING GAME----')
            for i in range(4):
                self.get_gameplay_screen()
            return self.frames_to_tensor()

    def is_play_button(self):
        region = {"top": 950, "left": 1000, "width": 250 , "height": 70}
        area = self.crop_screen(self.grab_screen(), region)
        area = cv2.cvtColor(area, cv2.COLOR_BGRA2GRAY)
        similarity = cv2.matchTemplate(area, self.play_button_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(similarity)
        if max_val > 0.97:
            return True
        else:
            return False

    def is_start_text(self):
        region = {"top": 780, "left": 590, "width": 510, "height": 120}
        area = self.crop_screen(self.grab_screen(), region)
        area = cv2.cvtColor(area, cv2.COLOR_BGRA2GRAY)
        similarity = cv2.matchTemplate(area, self.play_text, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(similarity)
        if max_val > 0.7:
            return True
        else:
            return False

    def step(self, action):
        self.do_action(action)
        time.sleep(0.05)
        self.get_gameplay_screen()

        reward = 0
        done = self.is_done()
        if done:
            reward -= 10
            self.score = self.get_score()
            time.sleep(4.8)
            self.game_state = "MENU"
        else:
            reward += 0.5
        observation = self.frames_to_tensor()
        return observation, reward, done

    def grab_screen(self):
        monitor = {"top": 0, "left": 0, "width": 1920 , "height": 1080}
        with mss.MSS() as sct:
            screenshot = sct.grab(monitor)
            return np.array(screenshot)

    def crop_screen(self, frame, r):
        return frame[r['top']: r['top'] + r['height'], r['left']: r['left'] + r['width']]

    def get_gameplay_screen(self, window_width=84, window_height=84):
        gameplay = self.crop_screen(self.grab_screen(), self.gameplay_region)
        gameplay = cv2.cvtColor(gameplay, cv2.COLOR_BGRA2GRAY)
        gameplay = cv2.resize(gameplay, (window_width, window_height))
        self.frames.append(gameplay)

    def frames_to_tensor(self):
        state = np.array(self.frames, dtype=np.float32) / 255.0
        return torch.tensor(state, dtype=torch.float32)

    def is_done(self):
        crop = self.crop_screen(self.grab_screen(), self.pause_region)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        similarity = cv2.matchTemplate(crop, self.pause_button, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(similarity)
        if max_val > 0.85:
            return False
        else:
            return True

    def do_action(self, choice):
        action_table = {
            'UP' : 'up',
            'DOWN' : 'down',
            'LEFT' : 'left',
            'RIGHT' : 'right',
            'NONE' : None
        }
        key = action_table[choice]
        if key is not None:
            return pyautogui.press(key)

    def get_score(self):
        crop = self.crop_screen(self.grab_screen(), self.score_region)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        _, crop = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
        result = self.reader.readtext(crop, allowlist='0123456789', detail=0)
        if not result:
            return 0
        try:
            score = int(result[0])
            return score
        except ValueError:
            return 0

