import cv2
import numpy as np
from scipy import stats

def find_threshold_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply thresholding to convert grayscale to binary image
    ret,thresh = cv2.threshold(gray,1,255,0)
    return thresh

def draw_points(img, pnts, out_file):
    for pnt in pnts:
        img_out = cv2.circle(img, tuple(pnt), 10, (255,0,0), -1)
    if out_file is not None:
        cv2.imwrite(out_file, img_out)
    return img_out

""" Rule: top-left means we priotize top side over left side """
def get_four_corners(img):    
    num_pixels = 500
    num_points = 20
    threshold1 = 10
    threshold2 = 30
    
    def get_top_left_points(img):
        ret =[]
        for x in range(num_pixels):
            for y in range(num_pixels):
                if sum(img[y,x]) > threshold1:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    def get_left_top_points(img):
        ret =[]
        for y in range(num_pixels):
            for y in range(num_pixels):
                if sum(img[y,x]) > threshold1:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    def get_bottom_left_points(img):
        ret =[]
        # check the first 500x500 pixels
        for x in range(num_pixels):
            for y in range(img.shape[0]-1, img.shape[0]-1-num_pixels, -1):
                if sum(img[y,x]) > threshold2:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    def get_top_right_points(img):
        ret =[]
        # check the first 500x500 pixels
        for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
            for y in range(num_pixels):
                if sum(img[y,x]) > 10:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    def get_bottom_right_points(img):
        ret =[]
        # check the first 500x500 pixels
        for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
            for y in range(img.shape[0]-1, img.shape[0]-1-num_pixels, -1):
                if sum(img[y,x]) > 10:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    def get_left_bottom_points(img):
        num_points = 10
        ret =[]
        # check the first 500x500 pixels
        for y in range(int(img.shape[0]/2-num_pixels/2), int(img.shape[0]/2+num_pixels/2), 5):
            for x in range(int(img.shape[1]/2)):
                if sum(img[y,x]) > 30:
                    ret.append([x,y])
                    if len(ret) == num_points:
                        return np.array(ret)
                    else:
                        break
    
    tl_pnts = get_top_left_points(img)
    slope, intercept, _, _, _ = stats.linregress(tl_pnts[:, 0], tl_pnts[:,1])
    top_left = np.array([0, int(np.floor(intercept))])
    
    tr_pnts = get_top_right_points(img)
    slope, intercept, _, _, _ = stats.linregress(tr_pnts[:, 0], tr_pnts[:,1])
    x = img.shape[1] - 1
    y = int(np.floor(slope * x + intercept))
    top_right = np.array([x, y])
    
    br_pnts = get_bottom_right_points(img)
    slope, intercept, _, _, _ = stats.linregress(br_pnts[:, 0], br_pnts[:,1])
    x = img.shape[1] - 1
    y = int(np.floor(slope * x + intercept))
    bottom_right = np.array([x, y])
    
    lb_pnts = get_left_bottom_points(img)
    slope, intercept, _, _, _ = stats.linregress(lb_pnts[:, 0], lb_pnts[:,1])
    y = img.shape[0] - 1
    x = int(np.floor((y-intercept)/slope))
    left_bottom = np.array([x, y])
    return [top_left, top_right, bottom_right, left_bottom]

if __name__ == '__main__':
    img = cv2.imread('out_without_crop_1.jpg')
    [top_left, top_right, bottom_right, bottom_left] = get_four_corners(img)
    draw_points(img, [top_left, top_right, bottom_right, bottom_left], 'test.jpg')

