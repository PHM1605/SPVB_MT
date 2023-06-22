import cv2
import numpy as np

def black_percentage(img, mode='color'):
    """ 
    Count percentage of black pixels
    Args:
        mode: 'color', 'gray', 'binary'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    area = thresh.shape[0] * thresh.shape[1]
    black_count = area - cv2.countNonZero(thresh)
    return np.round(black_count / area * 100, 2)

if __name__ == '__main__':
    img = cv2.imread('data/output/out.png')
    pct = black_percentage(img)
    print(f"Black percentage is {pct}%")