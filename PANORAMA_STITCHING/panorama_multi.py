import cv2
import numpy as np

##read image 1, resize it and convert to grayscale image
image_1_resize= cv2.imread('I.jpeg')
image_1_resize = cv2.resize(image_1_resize, (300, 500))
image_1= cv2.cvtColor(image_1_resize,cv2.COLOR_BGR2GRAY)
##read image 2, resize it and convert to grayscale image
image_2_resize = cv2.imread('II.jpeg')
image_2_resize = cv2.resize(image_2_resize,(300, 500))
image_2 = cv2.cvtColor(image_2_resize,cv2.COLOR_BGR2GRAY)
##read image 3, resize it and convert to grayscale image
image_3_resize = cv2.imread('III.jpeg')
image_3_resize = cv2.resize(image_3_resize, (300, 500))
image_3 = cv2.cvtColor(image_3_resize, cv2.COLOR_BGR2GRAY)
cv2.imwrite("3/gray_1.jpg", image_1)
cv2.imwrite("3/gray_2.jpg", image_2)
cv2.imwrite("3/gray_3.jpg", image_3)
sift = cv2.xfeatures2d.SURF_create()
##find the key points  with SIFT
kp_12, des_12 = sift.detectAndCompute(image_1,None)
kp_2, des_2 = sift.detectAndCompute(image_2,None)

##match features
match_12 = cv2.BFMatcher()
match_12 = match_12.knnMatch(des_12,des_2,k=2)

good = []
for m,n in match_12:
    if m.distance < 0.9*n.distance:
        good.append(m)

#draw green line to show same feature detected
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
image_draw_line = cv2.drawMatches(image_1, kp_12, image_2, kp_2, good, None, **draw_params)
cv2.imwrite("3/feature_extrect_from_image_1&2.jpg",image_draw_line)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts_12 = np.float32([ kp_12[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts_12 = np.float32([ kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M1 , mask = cv2.findHomography(src_pts_12, dst_pts_12, cv2.RANSAC,15.0)
    h1,w1 = image_1.shape
    pts_12 = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
    dst_12 = cv2.perspectiveTransform(pts_12,M1)
    image_12 = cv2.polylines(image_2,[np.int32(dst_12)],True,255,3, cv2.LINE_AA)
    cv2.imwrite("3/image_1_&_2_overlapping.jpg",image_12)
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))


dst = cv2.warpPerspective(image_1_resize,M1,(image_2_resize.shape[1] + image_1_resize.shape[1], image_2_resize.shape[0]))
cv2.imwrite("3/1&2_image_stitched.jpg", dst)
dst[0:image_2_resize.shape[0], 0:image_2_resize.shape[1]] = image_2_resize




##to crop out extra black screen, when we stitch 2  image of 300x300 images the final output image is of 600x300;
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

image_12 = trim(dst)
image_12_final= image_12
cv2.imwrite("3/trimmed_12.jpg",image_12_final)

image_12 = cv2.cvtColor(image_12,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SURF_create()
kp_12, des_12 = sift.detectAndCompute(image_12,None)
kp_123, des_123 = sift.detectAndCompute(image_3,None)

match_123 = cv2.BFMatcher()
match_123 = match_123.knnMatch(des_12,des_123,k=2)


good = []
for m,n in match_123:
    if m.distance < 0.9*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=2)

image_123_draw_line = cv2.drawMatches(image_12_final, kp_12, image_3_resize, kp_123, good, None, **draw_params)
cv2.imwrite("3/feature_extrect_from_image_1&2&3.jpg",image_123_draw_line)


MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts_23 = np.float32([ kp_12[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts_23 = np.float32([ kp_123[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M2 , mask = cv2.findHomography(src_pts_23, dst_pts_23, cv2.RANSAC,15.0)
    h2,w2 = image_1.shape
    pts_23 = np.float32([ [0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0] ]).reshape(-1,1,2)
    dst_23 = cv2.perspectiveTransform(pts_23, M2)
    image_123 = cv2.polylines(image_123_draw_line,[np.int32(dst_23)],True,255,3, cv2.LINE_AA)
    #cv2.imwrite("3/image_1_&_2_&_3_overlapping.jpg",image_123)
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))


dst = cv2.warpPerspective(image_12_final, M2,(image_12_final.shape[1] + image_3_resize.shape[1], image_12_final.shape[0]))
cv2.imwrite("3/1&2&3_image_stitched.jpg", dst)
dst[0:image_3_resize.shape[0], 0:image_3_resize.shape[1]] = image_3_resize

cv2.imshow("final", trim(dst))
cv2.imwrite("3/final.jpg",trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows()