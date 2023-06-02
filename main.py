import argparse, cv2, imutils
from imutils import paths
import numpy as np

images_folder = 'data/images'
output_file = 'output.jpg'

print('[INFO] loading images...')
imagePaths = sorted(list(paths.list_images(images_folder)))
images = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)
    
print('[INFO] stitching images...')
stitcher = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
stitcher.setPanoConfidenceThresh(0.0) # might be too aggressive for real examples
(status, stitched) = stitcher.stitch(images)
assert status == 0
cv2.imwrite(output_file, stitched)
# cv2.imshow("Stitched", stitched)
# cv2.waitKey(0)
