import cv2
import numpy as np
from math import sqrt
from math import pow


def KanadeByList(listData):
    feature_params = dict(maxCorners=500,
                       qualityLevel=0.5,
                       minDistance=1,
                       blockSize=7)

    lk_params = dict(winSize=(10, 10),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    old_frame = cv2.imread('Data/' + listData[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)
    width, height = old_frame.shape[:2]

    for element in listData:
        print('Traitement par Kanade de : Data/' + element + ' ==> DataKanade/Kanade_' + element)

        cv2.imshow('fenetre', old_gray)
        cv2.waitKey(100)

        frame = cv2.imread('Data/' + element)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        startX = 0
        startY = 0
        endX = width
        endY = height

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            dist = sqrt(pow(a-c, 2) + pow(b-d, 2))
            if dist > 0.5:

                if a > startX:
                    startX = a
                if b > startY:
                    startY = b
                if a < endX:
                    endX = c
                if b < endY:
                    endY = d

                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                print("dist: " + str(sqrt((pow(a-c, 2) + pow(b-d, 2)))))
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

        cv2.imwrite('DataKanade/Kanade_' + element, img)
        old_gray = frame_gray
#        p0 = good_new.reshape(-1, 1, 2)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
