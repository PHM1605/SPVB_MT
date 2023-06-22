# https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from blur_detection import blur_detection

def sort_imgs_str(img_names):
    return sorted(img_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, help = 'path to input directory of images to stitch', default="images")
ap.add_argument("-o", "--output", type=str, help = "path to the output image", default="output2.png")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["images"]))
imagePaths = sort_imgs_str(imagePaths)
images = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

# # blur filtering
# ret1, ret2 = [], []
# for i, img in enumerate(images):
#     if not blur_detection(img): 
#         ret1.append(img)
#         ret2.append(imagePaths[i])
# images, imagePaths = ret1, ret2

print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()

batch = 3
for i in range(0, len(images), batch):
    subset = images[i:(i+batch)]
    (status, stitched) = stitcher.stitch(subset)
    if status == 0:
        cv2.imwrite(f"output{i}.png", stitched)
    else:
        print("[INFO] image stitching failed ({} - {})".format(i, status))