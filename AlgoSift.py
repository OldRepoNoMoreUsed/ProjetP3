import cv2


def CreateSiftImg(dataName):
    print('Traitement par Sift de : Data/' + dataName + ' ==> DataSift/Sift_' + dataName)
    img = cv2.imread('Data/' + dataName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    cv2.imwrite('DataSift/Sift_' + dataName, img)


def CreateSiftImgByList(list):
    for element in list:
        print('Traitement par Sift de : Data/' + element + ' ==> DataSift/Sift_' + element)
        img = cv2.imread('Data/' + element)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv2.drawKeypoints(gray, kp, img)
        cv2.imwrite('DataSift/Sift_' + element, img)
