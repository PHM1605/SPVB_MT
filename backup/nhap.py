import numpy as np
import cv2, glob, imutils, os, re

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
        
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isv3:
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        # Convert keypoints from Keypoint objects to Numpy arrays
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        pass
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m)==2 and m[0].distance<m[1].distance*ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for(_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None
    
    def remove_black_region(self, img):
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)[1]
        contours_orig = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours = contours_orig[0]
        contours = contours.squeeze(1)
        (x0, y0) = int(min(contours[:,0])), int(min(contours[:,1]))
        (x1, y1) = int(max(contours[:,0])), int(max(contours[:,1]))
        thresh = thresh[y0:y1+1, x0:x1+1]
        img = img[y0:y1+1, x0:x1+1]
        contours[:,0] = contours[:,0] - x0 
        contours[:,1] = contours[:,1] - y0
        i = 0
        for cnt in contours:
            x, y = cnt[0], cnt[1]
            if x > 0 and x < thresh.shape[1]-1 and y > 0 and y < thresh.shape[0]-1: # contours inside image
                distance = {'left': x, 
                            'right': (thresh.shape[1] - 1 - x), 
                            'top': y, 
                            'bottom': (thresh.shape[0] - 1 - y)}
                min_index = np.argmin([distance[dist] for dist in distance])
                mode = list(distance.keys())[min_index]
                #cv2.imwrite(f"data/output_before_{i}.png", img)
                # img = cv2.circle(img, (x,y), 3, (0,255,0), 2)
                # if np.array_equiv(thresh[y-1, x], [0,0,0]) and not np.array_equiv(thresh[y+1, x], [0,0,0]):
                #     img[:y, x] = img[y+1, x]
                # elif np.array_equiv(thresh[y, x+1], [0,0,0]) and not np.array_equiv(thresh[y, x-1], [0,0,0]):
                #     img[y, x:] = img[y, x-1]
                # elif np.array_equiv(thresh[y+1, x], [0,0,0]) and not np.array_equiv(thresh[y-1, x], [0,0,0]):
                    
                #     img[y:, x] = img[y-1, x]
                # elif np.array_equiv(thresh[y, x-1], [0,0,0]) and not np.array_equiv(thresh[y, x+1], [0,0,0]):
                #     img[y, :x] = img[y, x+1]
                if mode == 'top':
                    img[:y, x] = img[y+1, x]
                elif mode == 'right':
                    img[y, x:] = img[y, x-1]
                elif mode == 'bottom':                    
                    img[y:, x] = img[y-1, x]
                elif mode == 'left':
                    img[y, :x] = img[y, x+1]
                #cv2.imwrite(f"data/output_after_{i}.png", img)
                i += 1
                
        return img
        
        
    
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        # Detect keypoints and extract local invariant descriptors
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None
        (matches, H, status) = M
        img = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        img[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        
        if showMatches:
            pass
        
        # Filter out the black region
        cv2.imwrite(f"data/output_before_{i}.png", img)
        img = self.remove_black_region(img)
        cv2.imwrite(f"data/output_after_{i}.png", img)
        return img
                

images_folder = 'data/images'
output_file = 'output.jpg'

print('[INFO] loading images...')
images_path = glob.glob( os.path.join('data/images', '*.jpg'))
images_path = sorted(images_path, key=lambda x:float(re.findall("([0-9]+?)\.jpg",x)[0]))
images = []
for imagePath in images_path:
    image = cv2.imread(imagePath)
    images.append(image)

img_kept = images[0]
stitcher = Stitcher()
for i in range(1, len(images)):
    img_warped = images[i]
    img_kept = stitcher.stitch([img_kept, img_warped], showMatches = False) 
    
print('Done')
