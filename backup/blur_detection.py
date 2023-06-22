import cv2

def blur_detection(img, thres = 120):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(grey, cv2.CV_64F).var()
    if var < thres:
        return True
    return False

if __name__ == '__main__':
    img = cv2.imread('data/images/frame_36.jpg')
    
    
    if blur_detection(img):
        print('Image is Blurred')
    else:
        print('Image Not Blurred')

    