#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import concurrent.futures
import math
import tkinter as tk
from tkinter.font import Font

import cv2

from utils import load_cascade


CASCADE = '/path/to/face/cascade'


def save_image(fp, img):
    # TODO: add variance check with Laplacian filter to discard too blurry images
    cv2.imwrite(fp, img)


class SelectionWindow(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('EyeTracker Data Collector')
        self.eye_side = tk.StringVar()
        self.eye_side.set(None)
        self.eye_state = tk.StringVar()
        self.eye_state.set(None)
        self._set_up_content()

    def get_values(self):
        return self.eye_side.get(), self.eye_state.get()

    def _set_up_content(self):
        title_font = Font(size=12, weight='bold', underline=1)
        title = tk.Label(self, text='Options', font=title_font)
        title.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=5)

        header_font = Font(size=10, weight='bold')
        side_of_face = tk.Label(self, text='Eye:', font=header_font)
        side_of_face.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        eye_side_options = ['Both', 'Left', 'Right']
        for row, side in enumerate(eye_side_options):
            radio = tk.Radiobutton(self, text=side, variable=self.eye_side, value=side)
            radio.grid(row=2+row, column=0, padx=5, pady=2, sticky='w')

        state_of_face = tk.Label(self, text='State:', font=header_font)
        state_of_face.grid(row=1, column=1, sticky='w', padx=5, pady=5)

        eye_state_options = ['Open', 'Closed']
        for row, state in enumerate(eye_state_options):
            radio = tk.Radiobutton(self, text=state, variable=self.eye_state, value=state)
            radio.grid(row=2+row, column=1, padx=5, pady=2, sticky='w')

        note = tk.Label(self, text='After launching, to stop recording,\npress Esc')
        note.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ok = tk.Button(self, text='OK', command=self._close, bg='#cccccc')
        ok.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='we')

    def _close(self):
        self.destroy()


class Capture:

    def __init__(self, eye_to_record, eye_state, width, height):
        self.eye_to_record = eye_to_record
        self.eye_state = eye_state
        self.width = width
        self.height = height

        # prepare the data storage location
        self._prepare_data_destination()

        # load the cascade
        self.cascade = load_cascade(CASCADE)

        # acquire video capturing device
        self.v_cap = cv2.VideoCapture(0)

    def _prepare_data_destination(self):
        os.makedirs('./data/train/open', exist_ok=True)
        os.makedirs('./data/train/closed', exist_ok=True)

    def launch(self, delay=5):
        assert isinstance(delay, int)

        if delay > 10:
            delay = 10

        # initial values
        start_time = None
        capture = False
        color = (0, 255, 0)

        # for simplicity reasons we use an fixed offset
        offset = .15

        # starting the image capturing, due to image saving we apply a thread pool
        with concurrent.futures.ThreadPoolExecutor() as threads:
            while True:
                ret, img = self.v_cap.read()
                img = cv2.flip(img, 1)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_list = self.cascade.detectMultiScale(gray, 1.3, 5)

                if start_time is None:
                    start_time = time.time()
                number = math.ceil(delay - (time.time() - start_time))
                if number <= 0:
                    color = (0, 0, 255)
                    capture = True

                for (x, y, w, h) in face_list:
                    y += int(h*offset)
                    cv2.rectangle(img, (x, y), (x + w//2, y + h//2), color, 2)
                    cv2.rectangle(img, (x + w//2, y), (x + w, y + h // 2), color, 2)

                    if capture:
                        time_stamp = str(time.time()).replace('.', '')
                        try:
                            if self.eye_to_record in ['Both', 'Left']:
                                left = gray[y:y + h//2, x:x + w//2].copy()
                                left = cv2.resize(left, (self.width, self.height))
                                threads.submit(
                                    save_image,
                                    os.path.join(
                                        './data/train',
                                        self.eye_state.lower(),
                                        'l_' + '{:0<17}'.format(time_stamp) + '.bmp'
                                    ),
                                    left
                                )
                            if self.eye_to_record in ['Both', 'Right']:
                                right = gray[y:y + h//2, x + w//2:x + w].copy()
                                right = cv2.resize(right, (self.width, self.height))
                                right = cv2.flip(right, 1)
                                threads.submit(
                                    save_image,
                                    os.path.join(
                                        './data/train',
                                        self.eye_state.lower(),
                                        'r_' + '{:0<17}'.format(time_stamp) + '.bmp'
                                    ),
                                    right
                                )
                        except Exception as err:
                            print(str(err))
                            return -1

                if not capture:
                    cv2.putText(img, str(number), (int(img.shape[1] / 2 * 0.85), int(img.shape[0] / 2 * 1.2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

                cv2.imshow("Live image", img)

                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

        self.v_cap.release()
        cv2.destroyAllWindows()


def main():
    sw = SelectionWindow()
    sw.mainloop()

    eye_to_record, eye_state = sw.get_values()

    if eye_to_record == "None":
        print('You did not select an eye, quitting...')
        return -1

    if eye_state == "None":
        print('You did not select an eye state, quitting...')
        return -1

    print(f'Selected: {eye_to_record}, {eye_state}')

    capture = Capture(eye_to_record, eye_state, 450, 450)
    capture.launch()

    return 0


if __name__ == '__main__':
    main()
