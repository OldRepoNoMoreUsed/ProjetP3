import cv2
import numpy as np
import time


def GrimsonByList(listData):
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    kernel = np.ones((3,3), np.uint8)
    detector = cv2.SimpleBlobDetector()

    for element in listData:
        print('Traitement par Grimson de : Data/' + element + ' ==> DataGrimson/Grimson_' + element)
        frame = cv2.imread('Data/' + element)

        fgmask = fgbg.apply(frame)
        time.sleep(0.2)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        keypoints = detector.detect(fgmask)
        im_with_keypoints = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        for cnt in contours:
            if 100 >= cv2.contourArea(cnt) >= 0:
                (x, y, w, h) = cv2.boundingRect(cnt)
                #points = cv2.boxPoints(rect)
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.drawContours(frame, contours, -1, (0,255,0), 3)

        cv2.imshow('frame', im_with_keypoints)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

        cv2.imwrite('DataGrimson/Grimson_' + element, fgmask)
