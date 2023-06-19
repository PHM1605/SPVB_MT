import cv2, glob, imutils
from find_four_corners import find_four_corners
from utils import get_four_corners
from perspective_transform import perspective_transform
from utils import find_threshold_img
import numpy as np
import matplotlib.pyplot as plt

def sort_imgs_str(img_names):
    return sorted(img_names, key=lambda x: int(x.split('.')[0].rsplit('frame')[-1]))

# remove matches with too high slope
def filter_matches(matches, kp1, kp2):
    match_ratio = 0.6
    valid_matches = []
    for m1, m2 in matches:
        if m1.distance < match_ratio * m2.distance:
            valid_matches.append(m1)
    # ret = []
    # for i, match in enumerate(valid_matches):
    #     p1 = kp1[match.queryIdx].pt
    #     p2 = kp2[match.trainIdx].pt
    #     if abs(p1[1] - p2[1]) < 100:
    #         ret.append(match)
    ret = valid_matches
    return ret

def filter_contours(contours):
    contours_size = [c.shape[0] for c in contours]
    chosen_contour = contours[np.argmax(contours_size)]
    contours = chosen_contour.squeeze(1)
    return contours

def stitch(prev, curr):
    sift = cv2.SIFT_create()
    h1, w1 = prev.shape[0:2]
    h2, w2 = curr.shape[0:2]
    prev_crop = prev[:, w1-w2:]
    diff = np.size(prev, axis=1) - np.size(prev_crop, axis = 1)
    kp1, des1 = sift.detectAndCompute(prev_crop, None)
    kp2, des2 = sift.detectAndCompute(curr, None)
    bf = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    valid_matches = filter_matches(matches, kp1, kp2)
            
    # Extract the coordinates of matching points
    img1_pts = []
    img2_pts = []
    for match in valid_matches:
        img1_pts.append(kp1[match.queryIdx].pt)
        img2_pts.append(kp2[match.trainIdx].pt)
        
    # # Draw matches
    # matched_image = cv2.drawMatches(prev, kp1, curr, kp2, valid_matches, None, flags=2)
    # cv2.imwrite('a.jpg', matched_image)
    
    # Formalize as matrices (for the sake of computing Homography)
    img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
    img1_pts[:, :, 0] += diff  # Recover its original coordinates
    img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
    
    # Compute the Homography matrix
    H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
    
    # Recompute the Homography in order to improve robustness
    for i in range(mask.shape[0] - 1, -1, -1):
        if mask[i] == 0:
            np.delete(img1_pts, [i], axis=0)
            np.delete(img2_pts, [i], axis=0)
    
    H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
    stitched_img = stitch_by_H(prev, curr, H)
    return stitched_img

def crop_black(img):
    """Crop off the black edges."""
    thresh = find_threshold_img(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    best_rect = (0, 0, 0, 0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        deltaHeight = h - y
        deltaWidth = w - x
        area = deltaHeight * deltaWidth
        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)
    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
                   best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        img_crop = img
    """ Crop off the inclined edges."""
    img_crop = cv2.copyMakeBorder(img_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    thresh = find_threshold_img(img_crop)
    contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours)
    four_corners = find_four_corners(contours, img_crop)
    img_crop = perspective_transform(img_crop, four_corners)
    #cv2.imwrite('crop.jpg', img_crop)
    thresh = find_threshold_img(img_crop)
    contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0].squeeze(1)
    contours = np.array([c for c in contours  if c[0]!=0 and c[0]!=img_crop.shape[1]-1 and c[1]!=0 and c[1]!=img_crop.shape[0]-1])
    upper_index = [c[1] for c in contours if c[1]<img_crop.shape[0]/2]
    upper_index = max(upper_index) if len(upper_index) > 0 else 0
    lower_index = [c[1] for c in contours if c[1]>img_crop.shape[0]/2]
    lower_index = min(lower_index) if len(lower_index) > 0 else img.shape[0]
    img_crop = img_crop[upper_index:lower_index]

    return img_crop

def crop_black_2(img):
    """Crop off the black edges."""
    thresh = find_threshold_img(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    best_rect = (0, 0, 0, 0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        deltaHeight = h - y
        deltaWidth = w - x
        area = deltaHeight * deltaWidth
        if area > max_area and deltaHeight > 0 and deltaWidth > 0:
            max_area = area
            best_rect = (x, y, w, h)
    if max_area > 0:
        img_crop = img[best_rect[1]:best_rect[1] + best_rect[3],
                   best_rect[0]:best_rect[0] + best_rect[2]]
    else:
        img_crop = img
        
    #img_crop = perspective_transform(img_crop, get_four_corners(img_crop, mode='left_bottom'))
    #img_crop = perspective_transform(img_crop, get_four_corners(img_crop, mode='bottom_left'))
    return img_crop
    
def stitch_by_H(img1, img2, H):
    """Use the key points to stitch the images.
    img1: the image containing frames that have been joint before.
    img2: the newly selected key frame.
    H: Homography matrix, usually from compute_homography(img1, img2).
    """
    # Get heights and widths of input images
    h1, w1 = img1.shape[0:2]
    h2, w2 = img2.shape[0:2]

    # Store the 4 ends of each original canvas
    img1_canvas_orig = np.float32([[0, 0], [0, h1],
                                   [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_canvas = np.float32([[0, 0], [0, h2],
                              [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # The 4 ends of (perspective) transformed img1
    img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

    # Get the coordinate range of output (0.5 is fixed for image completeness)
    output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
    [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

    # The output matrix after affine transformation
    transform_array = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0, 1]])

    # Warp the perspective of img1
    img_output = cv2.warpPerspective(img1, transform_array.dot(H),
                                     (x_max - x_min, y_max - y_min), flags=cv2.INTER_NEAREST)

    for i in range(-y_min, h2 - y_min):
        for j in range(-x_min, w2 - x_min):
            if np.any(img2[i + y_min][j + x_min]):
                img_output[i][j] = img2[i + y_min][j + x_min]
    return img_output

img_list = glob.glob('data/images/*.jpg')
img_list = sort_imgs_str(img_list)
batch = 100
batch_list = []
prev = cv2.imread(img_list[0])
for i in range(1, len(img_list), batch):
    if prev is None:
        prev = cv2.imread(img_list[i-1])
    elif i+batch > len(img_list):
        batch = i+batch-len(img_list)
    for j in range(0, batch, 5):
        curr = cv2.imread(img_list[i+j])
        stitched_img = stitch(prev, curr)
        crop_img = crop_black_2(stitched_img)
        print(f'Stitch frame{i+j-1} and frame{i+j} successfully')
        crop_img = imutils.resize(crop_img, height=1920)
        prev = crop_img
    crop_img = perspective_transform(crop_img, get_four_corners(crop_img, mode='left_bottom'))
    crop_img = perspective_transform(crop_img, get_four_corners(crop_img, mode='bottom_left'))
    cv2.imwrite(f'image_batch_{int(i/batch)}.jpg', crop_img)
    print(f'{int(i/batch)} batch completed')
    batch_list.append(crop_img)
    prev = None
    break

# prev = batch_list[0]
# curr = batch_list[1]
# final_img = stitch(prev, curr)
# for i, batch_img in enumerate(batch_list):
#     if i > 0:
#         curr = batch_list[i]
#         final_img = stitch(prev, curr)
#         prev = final_img
# final_img = perspective_transform(final_img, get_four_corners(final_img, mode='left_bottom'))
# final_img = perspective_transform(final_img, get_four_corners(final_img, mode='bottom_left'))
# cv2.imwrite('final.jpg', final_img)
# print('Stitching completed')

# img0 = cv2.imread('data/images/frame32.jpg')
# img1 = cv2.imread('data/images/frame37.jpg')
# # img0 = imutils.resize(img0, height=1920)
# # img1 = imutils.resize(img1, height=1920)
# test_img = stitch(img0, img1)
# test_img = perspective_transform(test_img, get_four_corners(test_img, mode='left_bottom'))
# test_img = perspective_transform(test_img, get_four_corners(test_img, mode='bottom_left'))
cv2.imwrite('test.png', crop_img)
# sift = cv2.SIFT_create()
# h1, w1 = img0.shape[0:2]
# h2, w2 = img1.shape[0:2]
# prev_crop = img0[:, w1-w2:]
# diff = np.size(img0, axis=1) - np.size(prev_crop, axis = 1)
# kp1, des1 = sift.detectAndCompute(prev_crop, None)
# kp2, des2 = sift.detectAndCompute(img1, None)
# bf = cv2.BFMatcher(normType=cv2.NORM_L2)
# matches = bf.knnMatch(des1, des2, k=2)
# valid_matches = filter_matches(matches, kp1, kp2)
        
# # Extract the coordinates of matching points
# img1_pts = []
# img2_pts = []
# for match in valid_matches:
#     img1_pts.append(kp1[match.queryIdx].pt)
#     img2_pts.append(kp2[match.trainIdx].pt)
    