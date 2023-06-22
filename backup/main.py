# import argparse, cv2, imutils,os
# from imutils import paths
# from blur_detection import blur_detection
# from find_four_corners import find_four_corners
# from perspective_transform import perspective_transform
# import numpy as np

# def sort_imgs_str(img_names):
#     return sorted(img_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

# def stitch(images, stitcher):
#     (status, stitched) = stitcher.stitch(images)
#     if status != 0:
#         return status
#     corners = find_four_corners(stitched)
#     stitched = perspective_transform(stitched, corners)
#     return stitched
    
# print('[INFO] loading images...')
# imagePaths = sort_imgs_str(list(paths.list_images('data/images')))
# images = []
# for imagePath in imagePaths:
#     image = cv2.imread(imagePath)
#     images.append(image)

# ret1, ret2 = [], []
# for i, img in enumerate(images):
#     if not blur_detection(img): 
#         ret1.append(img)
#         ret2.append(imagePaths[i])
# images, imagePaths = ret1, ret2

# print('[INFO] stitching images...')
# stitcher = cv2.Stitcher.create()
# stitcher = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
# num_images = 5
# for i in range(0, len(images)-num_images, num_images):
#     subset = images[i:i+num_images]
#     stitched = stitch(subset)
#     if not isinstance(stitched, int):
#         cv2.imwrite(f'data/output/final_out{i}.jpg', stitched)
#         print(f'Stitching frame{i} successful')
#     else:
#         print(f'Stitching frame{i} not successful, error {stitched}')
#         break

from torch_snippets import *
#subset= [images[10], images[11]]
#(status, img) = stitcher.stitch(subset)
img1 = 'data/images/frame_25.jpg'
img2 = 'data/images/frame_26.jpg'
queryImg = read(img1, 1)
queryImg_gray = read(img1)
trainImg = read(img2, 1)
trainImg_gray = read(img2)
subplots([queryImg, trainImg], nc=2, figsize=(10,5), titles=['QueryImg', 'Training image (Image to be stitched to Query image)'])

descriptor = cv2.ORB_create()
kpsA, featuresA = descriptor.detectAndCompute(trainImg_gray, None)
kpsB, featuresB = descriptor.detectAndCompute(queryImg_gray, None)
img_kpsA = cv2.drawKeypoints(trainImg_gray, kpsA, None, color=(0,255,0))
img_kpsB = cv2.drawKeypoints(queryImg_gray, kpsB, None, color=(0,255,0))
#subplots([img_kpsB, img_kpsA], nc=2, figsize=(10,5), titles=['Query image with keypoints', 'Training image with keypoints'])
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

best_matches = bf.knnMatch (featuresA,featuresB,k=2)
good_matches = []
for m1, m2 in best_matches:
    if m1.distance < 0.6*m2.distance:
        good_matches.append(m1)
matches = sorted(good_matches, key=lambda x : x.distance)

#best_matches = bf.match(featuresA, featuresB)
#matches = sorted(best_matches, key=lambda x : x.distance)

img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:200], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show(img3)
kpsA = np.float32([kp.pt for kp in kpsA])
kpsB = np.float32([kp.pt for kp in kpsB])
ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4)

width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]
result = cv2.warpPerspective(trainImg, H, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
_x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
_y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]
show(result[:_y,:_x])
#show(result)

#https://datahacker.rs/feature-matching-methods-comparison-in-opencv/

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img1 = cv2.imread('data/images/frame_21.jpg')
# img2 = cv2.imread('data/images/frame_20.jpg')
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
# bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# single_match = matches[0]
# single_match.distance
# matches = sorted(matches,key=lambda x:x.distance)
# #ORB_matches =cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
# #cv2.imwrite('test.jpg',ORB_matches)

# kpsA = np.float32([kp.pt for kp in keypoints1])
# kpsB = np.float32([kp.pt for kp in keypoints2])
# ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
# ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
# (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4)

# width = img1.shape[1] + img2.shape[1]
# height = img1.shape[0] + img2.shape[0]
# result = cv2.warpPerspective(img1, H, (width, height))
# result[0:img2.shape[0], 0:img2.shape[1]] = img2
# _x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
# _y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]
# cv2.imwrite('test.jpg',result[:_y,:_x])


# sift = cv2.SIFT_create()
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# bf = cv2.BFMatcher()
# matches = bf.knnMatch (descriptors1, descriptors2,k=2)
# good_matches = []
# for m1, m2 in matches:
#   if m1.distance < 0.6*m2.distance:
#     good_matches.append([m1])
#SIFT_matches =cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
#cv2.imwrite('test.jpg',SIFT_matches)

# kpsA = np.float32([kp.pt for kp in keypoints1])
# kpsB = np.float32([kp.pt for kp in keypoints2])
# ptsA = np.float32([kpsA[m[0].queryIdx] for m in good_matches])
# ptsB = np.float32([kpsB[m[0].trainIdx] for m in good_matches])
# (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4)

# width = img1.shape[1] + img2.shape[1]
# height = img1.shape[0] + img2.shape[0]
# result = cv2.warpPerspective(img1, H, (width, height))
# result[0:img2.shape[0], 0:img2.shape[1]] = img2
# _x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
# _y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]
#show(result[:_y,:_x])
#show(result)
# cv2.imwrite('test.jpg',result[:_y,:_x])


# for m1, m2 in matches:
#   if m1.distance < 0.6*m2.distance:
#     good_matches.append([m1])
    
# sift = cv2.SIFT_create()
# keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
# keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# FLAN_INDEX_KDTREE = 0
# index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
# search_params = dict (checks=50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch (descriptors1, descriptors2,k=2)
# good_matches = []
# for m1, m2 in matches:
#   if m1.distance < 0.5 * m2.distance:
#     good_matches.append([m1])
# flann_matches =cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)  
# cv2.imwrite('test.jpg',flann_matches)