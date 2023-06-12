import argparse, cv2, imutils,os
from imutils import paths
from blur_detection import blur_detection
from find_four_corners import find_four_corners
from perspective_transform import perspective_transform
import numpy as np

def sort_imgs_str(img_names):
    return sorted(img_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

def stitch(images):
    try:
        queryImg = images[0]
        queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
        trainImg = images[1]
        trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
    
        descriptor = cv2.ORB_create()
        kpsA, featuresA = descriptor.detectAndCompute(trainImg_gray, None)
        kpsB, featuresB = descriptor.detectAndCompute(queryImg_gray, None)
        # img_kpsA = cv2.drawKeypoints(trainImg_gray, kpsA, None, color=(0,255,0))
        # img_kpsB = cv2.drawKeypoints(queryImg_gray, kpsB, None, color=(0,255,0))
        # subplots([img_kpsB, img_kpsA], nc=2, figsize=(10,5), titles=['Query image with keypoints', 'Training image with keypoints'])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
        best_matches = bf.knnMatch (featuresA,featuresB,k=2)
        good_matches = []
        for m1, m2 in best_matches:
            if m1.distance < 0.4*m2.distance:
                good_matches.append(m1)
        matches = sorted(good_matches, key=lambda x : x.distance)
        print(len(matches))
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('data/output/draw_match.jpg', img3)
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])
        # ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        # ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        import hickle
        [ptsA, ptsB] = hickle.load('abc.hkl')
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4)
    
        width = trainImg.shape[1] + queryImg.shape[1]
        height = trainImg.shape[0] + queryImg.shape[0]
        result = cv2.warpPerspective(trainImg, H, (width, height))
        result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
        _x = np.nonzero(result.sum(0).sum(-1) == 0)[0][0]
        _y = np.nonzero(result.sum(1).sum(-1) == 0)[0][0]
        stitched = result[:_y,:_x]
        #corners = find_four_corners(stitched)
        #stitched = perspective_transform(stitched, corners)
        return stitched   
    except:
        return 1

    
print('[INFO] loading images...')
imagePaths = sort_imgs_str(list(paths.list_images('data/images')))
images = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (640, 480))
    images.append(image)

ret1, ret2 = [], []
for i, img in enumerate(images):
    if not blur_detection(img): 
        ret1.append(img)
        ret2.append(imagePaths[i])
images, imagePaths = ret1, ret2

print('[INFO] stitching images...')
num_images = 2
for i in range(0, len(images)-num_images, num_images):
    if i != 2:
        continue
    frame1_name = os.path.split(imagePaths[i+2])[-1]
    frame1_name = frame1_name.split('.')[0]
    frame2_name = os.path.split(imagePaths[i])[-1]
    frame2_name = frame2_name.split('.')[0]
    subset = images[i:i+num_images]
    stitched = stitch(subset)
    if not isinstance(stitched, int):
        cv2.imwrite(f'data/output/{frame1_name+"_"+frame2_name}.jpg', stitched)
        print(f'Stitching {frame1_name} and {frame2_name} successful')
    else:
        print(f'Stitching {frame1_name} and {frame2_name} not successful, error {stitched}')
        break
    
