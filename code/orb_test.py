import cv2 as cv

img = cv.imread('test2.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation

img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imwrite('orb_keypoints.jpg', img2)
