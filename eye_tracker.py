#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from tensorflow.keras.models import load_model

from utils import load_cascade


# define these and you're ready to go
MODEL = load_model('./model/my_cnn_model.tf')
CASCADE = '/path/to/face/cascade'
INPUT_SHAPE = (256, 256, 3)


def run_eye_tracker():
    try:
        # load the face cascade
        face_cascade = load_cascade(CASCADE)

        # prepare video capturing
        cap = cv2.VideoCapture(0)

        # for simplicity reasons, we use an fixed offset for the eye capturing
        offset = .15

        # labels and colors for both prediction results
        labels = {0: 'closed', 1: 'open'}
        colors = {0: (0, 255, 0), 1: (255, 0, 0)}

        # highlight eye locations with rectangles
        highlight = True

        while True:
            # reading one frame
            ret, img = cap.read()

            # pre processing of the image (frame)
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # get a list of faces in the image
            face_list = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face_list:
                # add offset to y
                y += int(h * offset)

                # crop and reshape eye locations
                left_eye = gray[y:y + h//2, x:x + w//2]
                left_eye = cv2.resize(left_eye, INPUT_SHAPE[:2])
                left_eye = left_eye.reshape((1,) + INPUT_SHAPE)

                right_eye = gray[y:y + h//2, x + w//2:x + w]
                right_eye = cv2.flip(right_eye, 1)
                right_eye = cv2.resize(right_eye, INPUT_SHAPE[:2])
                right_eye = right_eye.reshape((1,) + INPUT_SHAPE)

                # predict the eye states
                left_pred = MODEL.predict(left_eye)
                left_result = 0 if left_pred < 0.5 else 1
                # left_result = np.argmax(MODEL.predict(left_eye))
                right_pred = MODEL.predict(right_eye)
                right_result = 0 if right_pred < 0.5 else 1
                # right_result = np.argmax(MODEL.predict(right_eye))

                # update the texts according to the prediction results
                cv2.putText(
                    img, labels[left_result], (x-100, y+50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colors[left_result], 2, cv2.LINE_AA
                )
                cv2.putText(
                    img, labels[right_result], (x+w+10, y+50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, colors[right_result], 2, cv2.LINE_AA
                )

                if highlight:
                    cv2.rectangle(img, (x, y), (x + w//2, y + h//2), colors[left_result], 2)
                    cv2.rectangle(img, (x + w//2, y), (x + w, y + h//2), colors[right_result], 2)

            cv2.imshow('Eyetracker', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    except Exception as err:
        print(str(err))

    finally:
        cap.release()
        cv2.destroyAllWindows()
        return 0


if __name__ == '__main__':
    run_eye_tracker()
