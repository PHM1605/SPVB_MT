import cv2
import numpy as np

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

if __name__ == '__main__':
    img = cv2.imread('data/output/out.jpg')
    corners = [[361, 31],
               [1359, 77],
               [1741, 1391],
               [3, 1387]]
    out = perspective_transform(img, corners)
    cv2.imwrite('out_transformed.jpg', out)