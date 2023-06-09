import cv2
import numpy as np

def keep_corners(points, img):    
    distance = {}
    for i, pnt in enumerate(points):
        x = pnt[0]
        y = pnt[1]
        distance[i] = {
            'x': x,
            'y': y,
            'top_left': np.sqrt(x**2 + y**2),
            'top_right': np.sqrt((img.shape[1]-x)**2 + y**2),
            'bottom_right': np.sqrt((img.shape[1]-x)**2 + (img.shape[0]-y)**2),
            'bottom_left': np.sqrt(x**2 + (img.shape[0]-y)**2)}
    
    top_left = distance[np.argmin([distance[dt]['top_left'] for dt in distance])]
    top_right = distance[np.argmin([distance[dt]['top_right'] for dt in distance])]
    bottom_right = distance[np.argmin([distance[dt]['bottom_right'] for dt in distance])]
    bottom_left = distance[np.argmin([distance[dt]['bottom_left'] for dt in distance])]
    return np.array(
        [[top_left['x'], top_left['y']],
         [top_right['x'], top_right['y']],
         [bottom_right['x'], bottom_right['y']],
         [bottom_left['x'], bottom_left['y']]] )

img = cv2.imread('data/output/out.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
_, thres = cv2.threshold(gray, 1, 255,0)
thres = cv2.medianBlur(thres, 5)
#cv2.imwrite('tesst2.jpg', thres)
dst = cv2.cornerHarris(thres,5,3,0.04)
ret, dst = cv2.threshold(dst,0.2*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
corners = keep_corners(corners, img)
for i in range(len(corners)):
    img = cv2.circle(img, (int(corners[i,0]), int(corners[i,1])), 7, (0,255,0), 2)
cv2.imwrite('out.jpg', img)