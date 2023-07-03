import os, cv2, copy, imutils
import numpy as np
from scipy import stats
from PIL import ImageFont, ImageDraw, Image
import xml.etree.ElementTree as ET

class BoundingBox:
    def __init__(self, res, mode):
        if mode=='xyxy':
            self.x1 = res['x1']
            self.y1 = res['y1']
            self.x2 = res['x2']
            self.y2 = res['y2']
            self.cen_x = int((self.x1 + self.x2)/2)
            self.cen_y = int((self.y1 + self.y2)/2)
            self.w = int(self.x2 - self.x1)
            self.h = int(self.y2 - self.y1)
            
        elif mode=='xywh':
            self.cen_x = res['x']
            self.cen_y = res['y']
            self.w = res['width']
            self.h = res['height']
            self.x1 = int(res['x'] - res['width']/2)
            self.y1 = int(res['y'] - res['height']/2)
            self.x2 = int(res['x'] + res['width']/2)
            self.y2 = int(res['y'] + res['height']/2)

        self.prob = res['confidence']
        self.label = res['class']
        self.area = self.w * self.h

def calculate_overlap(box1, box2):
    a, b = box1, box2
    dx = min(a.x2, b.x2) - max(a.x1, b.x1)
    dy = min(a.y2, b.y2) - max(a.y1, b.y1)
    min_area = min(a.area, b.area)
    if dx >= 0 and dy >= 0:
        return dx * dy / min_area
    else:
        return 0

def remove_overlap_boxes(boxes, exclude_indices):
    flag = [True for _ in boxes]
    thres = 0.3  # new
    for i, box in enumerate(boxes):
        if flag[i] == False:
            continue
        for j, other_box in enumerate(boxes):
            if i in exclude_indices or j in exclude_indices or i == j:
                continue
            overlap_area = calculate_overlap(box, other_box)
            if overlap_area > thres:
                flag[i] = True
                flag[j] = False
    return [boxes[i] for i in range(len(boxes)) if flag[i]]

def sort_upper_to_lower(boxes, indices):
    list_boxes = [(idx, boxes[idx]) for idx in indices]
    list_boxes.sort(key=lambda shelf: shelf[1].y1)
    indices = [box[0] for box in list_boxes]
    return indices

# Vietnamese display on image
def put_text(img, text, loc):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(os.path.join("fonts", "SVN-Arial 2.ttf"), 14)
    draw.text(loc, text, font=font, fill=(0, 255, 255))
    return np.array(img_pil)


def draw_result(img, boxes, color, put_label, put_percent):
    ret = copy.deepcopy(img)
    for i, box in enumerate(boxes):
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret = cv2.rectangle(ret, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        if put_label:
            ret = cv2.putText(
                ret,
                box.label,
                (box.x1, box.y1 + 20),
                font,
                0.6,
                color,
                thickness,
            )
        if put_percent:
            ret = cv2.putText(
                ret,
                str(round(box.prob, 2)),
                (box.x1, box.y1),
                font,
                0.6,
                color,
                thickness,
            )
    return ret


# count number of items from a list of items
def count_item(item, list_items):
    count = 0
    for it in list_items:
        if item == it:
            count += 1
    return count


def search_bounding_boxes(boxes, label):
    ret = []
    for box in boxes:
        if box.label == label:
            ret.append([box.x1, box.y1, box.x2, box.y2])
    return ret



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
    edge_crop = 20
    
    sub_img = img.copy()
    sub_img = sub_img[:, edge_crop:img.shape[1]-edge_crop]
    thresh = find_threshold_img(sub_img)
    thresh_upper = thresh[: int(img.shape[0]/2), :]
    thresh_lower = thresh[int(img.shape[0]/2):, :]
    upper_limit = []
    lower_limit = []
    for x in range(thresh_upper.shape[1]):
        for y in range(thresh_upper.shape[0]):
            if thresh_upper[y,x] > 0:
                upper_limit.append(y)
                break
        for y in range(thresh_lower.shape[0]-1, 0, -1):
            if thresh_lower[y,x] > 0:
                lower_limit.append(y + int(img.shape[0]/2))
                break
    img = img[max(upper_limit):min(lower_limit), :]
    return img

def crop_edge(img):
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

def get_top_left(img, num_pixels=500, threshold=30):
    for x in range(num_pixels):
        for y in range(int(img.shape[0]/2)):
            if sum(img[y,x]) > threshold:
                return [x, y]
    return [0, 0]

def get_top_right(img, num_pixels=500, threshold=30):
    for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
        for y in range(img.shape[0]):
            if sum(img[y,x]) > threshold:
                return [x, y]
    return [img.shape[1]-1, 0]

def get_bottom_right(img, num_pixels=500, threshold=30):
    for x in range(img.shape[1]-1, img.shape[1]-1-num_pixels, -1):
        for y in range(img.shape[0]-1, 0, -1):
            if sum(img[y,x]) > threshold:
                return [x,y]
    return [img.shape[1]-1, img.shape[0]-1]

def get_left_bottom_points(img, num_pixels=500, threshold=30):
    num_points = 100
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
    return np.array(ret)
                
def get_bottom_left(img, num_pixels=500, threshold=30):
    for x in range(num_pixels):
        for y in range(img.shape[0]-1, img.shape[0]-1-num_pixels, -1):
            if sum(img[y,x]) > threshold:
                return [x,y]
    return [0, img.shape[0]-1]

""" Rule: top-left means we priotize top side over left side """
def get_four_corners(img, mode, num_pixels=500, threshold=30):    
    top_left = get_top_left(img, num_pixels, threshold)
    top_right = get_top_right(img, num_pixels, threshold)
    bottom_right = get_bottom_right(img, num_pixels, threshold)
    try:
        if mode == 'left_bottom':
            lb_pnts = get_left_bottom_points(img, num_pixels, threshold)
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

def get_slope(img, num_pixels=500, threshold=30):
    lb_pnts = get_left_bottom_points(img, num_pixels, threshold)
    slope, _, _, _, _ = stats.linregress(lb_pnts[:, 0], lb_pnts[:,1])
    return slope
    
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

def perspective_transform_and_resize(img, resize):
    curr_height = img.shape[0]
    img = perspective_transform(img, get_four_corners(img, mode='left_bottom'))
    img = perspective_transform(img, get_four_corners(img, mode='bottom_left'))
    if resize:
        img = imutils.resize(img, height=curr_height)
    return img

def sort_imgs_str(img_names):
    return sorted(img_names, key=lambda x: int(x.split('.')[0].rsplit('frame')[-1]))

def convert_xml_to_boxes(xml_file):
    tree = ET.parse(xml_file)
    objects = tree.findall('object')
    list_boxes = []
    for obj in objects:
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        label = obj.find('name').text
        box = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': label, 'confidence': 1.0}
        box = BoundingBox(box, mode='xyxy')
        list_boxes.append(box)
    return list_boxes

def convert_preds_to_boxes(preds, classes):
    list_boxes = []
    for pred in preds:
        box = {'x1': int(pred[0]), 'y1': int(pred[1]), 'x2': int(pred[2]), 'y2': int(pred[3]),\
               'class': classes[int(pred[5])], 'confidence': pred[4]}
        box = BoundingBox(box, mode='xyxy')
        list_boxes.append(box)
    return list_boxes

# group: SPVB/NON_SPVB; drink_type: CSD/ED/JD/TEA/WATER
def count_group_and_type(count_dict):
    groups, drink_types = [], []
    for key in count_dict.keys():
        key_split = key.split('_')
        group = key_split[0] if len(key_split) < 3 else key_split[0] + '_' + key_split[1]
        drink_type = key_split[-1]
        groups.append(group)
        drink_types.append(drink_type)
    groups = np.unique(groups)
    drink_types = np.unique(drink_types)
    
    ret_dict = {}
    for group in groups:
        ret_dict[group] = {}
        for drink_type in drink_types:
            ret_dict[group][drink_type] = count_dict[group+'_'+drink_type]
    return ret_dict

