import cv2, imutils
import numpy as np
from scipy import stats

def find_threshold_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert grayscale to binary image
    ret,thresh = cv2.threshold(gray,1,255,0)
    return thresh

def black_percentage(img, mode='color'):
    """ 
    Count percentage of black pixels
    Args:
        mode: 'color', 'gray', 'binary'
    """
    if mode == 'color':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    elif mode == 'gray':
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    else:
        thresh = img
    area = thresh.shape[0] * thresh.shape[1]
    black_count = area - cv2.countNonZero(thresh)
    return np.round(black_count / area * 100, 2)

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
        
    #img_crop = perspective_transform(img_crop, get_four_corners(img_crop, mode='left_bottom'))
    #img_crop = perspective_transform(img_crop, get_four_corners(img_crop, mode='bottom_left'))
    return img_crop

def draw_points(img, pnts, out_file):
    for pnt in pnts:
        img_out = cv2.circle(img, tuple(pnt), 10, (255,0,0), -1)
    if out_file is not None:
        cv2.imwrite(out_file, img_out)
    return img_out

""" Rule: top-left means we priotize top side over left side """
def get_four_corners(img, mode):    
    num_pixels = 500
    threshold = 30
    
    def get_top_left(img):
        for x in range(num_pixels):
            for y in range(int(img.shape[0]/2)):
                if sum(img[y,x]) > threshold:
                    return [x, y]
        return [0, 0]
    
    def get_top_right(img):
        for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
            for y in range(img.shape[0]):
                if sum(img[y,x]) > threshold:
                    return [x, y]
        return [img.shape[1]-1, 0]
    
    def get_bottom_right(img):
        for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
            for y in range(img.shape[0]-1, 0, -1):
                if sum(img[y,x]) > threshold:
                    return [x,y]
        return [img.shape[1]-1, img.shape[0]-1]
    
    def get_left_bottom_points(img):
        num_points = 10
        ret =[]
        # check the first 500x500 pixels
        for y in range(int(img.shape[0]/2-num_pixels/2), int(img.shape[0]/2+num_pixels/2), 5):
            for x in range(int(img.shape[1]/2)):
                if sum(img[y,x]) > threshold:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
                    
    def get_bottom_left(img):
        for x in range(num_pixels):
            for y in range(img.shape[0]-1, img.shape[0]-1-num_pixels, -1):
                if sum(img[y,x]) > threshold:
                    return [x,y]
        return [0, img.shape[0]-1]
    top_left = get_top_left(img)
    top_right = get_top_right(img)
    bottom_right = get_bottom_right(img)
    try:
        if mode == 'left_bottom':
            lb_pnts = get_left_bottom_points(img)
            slope, intercept, _, _, _ = stats.linregress(lb_pnts[:, 0], lb_pnts[:,1])
            y = img.shape[0] - 1
            x = int(np.floor((y-intercept)/slope))
            left_bottom = np.array([x, y])
            return [top_left, top_right, bottom_right, left_bottom]
        
        elif mode == 'bottom_left':
            bottom_left = get_bottom_left(img)
            
            return [top_left, top_right, bottom_right, bottom_left]
    except:
        return [top_left, top_right, bottom_right, np.array(0, img.shape[0]-1)]

def filter_matches(matches, kp1, kp2):
    match_ratio = 0.6
    valid_matches = []
    for m1, m2 in matches:
        if m1.distance < match_ratio * m2.distance:
            valid_matches.append(m1)
    ret = valid_matches
    return ret

def perspective_transform(img, corners):
    pt_A = corners[0] # top_left
    pt_B = corners[-1] # bottom_left
    pt_C = corners[2] # bottom_right
    pt_D = corners[1] # top_right
    width_AD = np.sqrt((pt_A[0] - pt_D[0])**2 + (pt_A[1] - pt_D[1])**2)
    width_BC = np.sqrt((pt_B[0] - pt_C[0])**2 + (pt_B[1] - pt_C[1])**2)
    max_width = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt((pt_A[0] - pt_B[0])**2 + (pt_A[1] - pt_B[1])**2)
    height_CD = np.sqrt((pt_C[0] - pt_D[0])**2 + (pt_C[1] - pt_D[1])**2)
    max_height = max(int(height_AB), int(height_CD))
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0,0], [0, max_height-1], [max_width-1,max_height - 1], [max_width-1, 0]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img, M, (max_width, max_height), flags = cv2.INTER_LINEAR)
    return out

def perspective_transform_and_resize(img):
    curr_height = img.shape[0]
    img = perspective_transform(img, get_four_corners(img, mode='left_bottom'))
    img = perspective_transform(img, get_four_corners(img, mode='bottom_left'))
    img = imutils.resize(img, height=curr_height)
    return img

def sort_imgs_str(img_names):
    return sorted(img_names, key=lambda x: int(x.split('.')[0].rsplit('frame')[-1]))

if __name__ == '__main__':
    img = cv2.imread('out_without_crop_1.jpg')
    [top_left, top_right, bottom_right, bottom_left] = get_four_corners(img, mode='bottom_left')
    draw_points(img, [top_left, top_right, bottom_right, bottom_left], 'test.jpg')

