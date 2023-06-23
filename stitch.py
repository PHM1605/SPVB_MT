import cv2, glob, imutils, os, shutil
import numpy as np
from utils import black_percentage, crop_black, crop_edge, get_four_corners, filter_matches
from utils import perspective_transform, perspective_transform_and_resize, sort_imgs_str

class StitchingClip():
    def __init__(self, clip_path):
        self.clip_path = clip_path
        self.frames_path = "data/images"
        self.output_path = "data/output"
        self.clip_name = os.path.basename(self.clip_path)
        self.folder_name = self.clip_name.split('.')[0]
        # if not os.path.exists(os.path.join(self.output_path, self.folder_name)):
        #     os.makedirs(os.path.join(self.output_path, self.folder_name))
        self.rewind = 6
        
    def extract_frames(self, rotate=None):
        if not os.path.exists(self.frames_path):
            os.makedirs(self.frames_path)
            
        if len(os.listdir(self.frames_path)) > 0:
            shutil.rmtree(self.frames_path)
            os.makedirs(self.frames_path)
            
            # user_input = input("Folder images is not empty -> erase all images to run (y/n?): ")
            # if user_input == 'y':
            #     shutil.rmtree(self.frames_path)
            #     os.makedirs(self.frames_path)
            # elif user_input == 'n':
            #     return

        vid_cap = cv2.VideoCapture(self.clip_path)
        sift = cv2.SIFT_create()
        success, last = vid_cap.read()
        if rotate is not None:
            last = cv2.rotate(last, rotate)
        cv2.imwrite('data/images/frame0.png', last)
        print("Captured frame0.png")
        count = 1
        frame_num = 1

        w = int(last.shape[1] * 2 / 3)  # the region to detect matching points
        stride = 30   # stride for accelerating capturing
        min_match_num = 50 # minimum number of matches required (to stitch well)
        max_match_num = 900  # maximum number of matches (to avoid redundant frames)
        image = None
        
        while success:
            if count % stride == 0:
                # Detect and compute key points and descriptors
                kp1, des1 = sift.detectAndCompute(last[:, -w:], None)
                kp2, des2 = sift.detectAndCompute(image[:, :w], None)
                bf = cv2.BFMatcher(normType=cv2.NORM_L2)  # Using Euclidean distance
                matches = bf.knnMatch(des1, des2, k=2)

                # Define valid match
                match_ratio = 0.8
                valid_matches = []
                for m1, m2 in matches:
                    if m1.distance < match_ratio * m2.distance:
                        valid_matches.append(m1)

                # At least 4 points are needed to compute Homography
                if len(valid_matches) > 4:
                    img1_pts = []
                    img2_pts = []
                    for match in valid_matches:
                        img1_pts.append(kp1[match.queryIdx].pt)
                        img2_pts.append(kp2[match.trainIdx].pt)

                    # Formalize as matrices (for the sake of computing Homography)
                    img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
                    img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

                    # Compute the Homography matrix
                    _, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

                    if min_match_num < np.count_nonzero(mask) < max_match_num:
                        last = image
                        print("Captured frame{}.png".format(frame_num))
                        cv2.imwrite('data/images/frame%d.png' % frame_num, last)
                        frame_num += 1
            success, image = vid_cap.read()
            if rotate is not None:
                image = cv2.rotate(image, rotate)
            count += 1

    def run(self):
        img_list = glob.glob('data/images/*.png')
        img_list = sort_imgs_str(img_list)
        
        prev = cv2.imread(img_list[0]) 
        crop_list = []
        for i in range(1, len(img_list)):
            curr = cv2.imread(img_list[i])
            stitched_img = self.stitch(prev, curr)
            #cv2.imwrite(f'test{i}.png', prev)
            crop_img = crop_edge(stitched_img)
            print(f'Stitch frame{i} and frame{i-1} successfully')
            crop_img = imutils.resize(crop_img, height=curr.shape[0])
            # print(black_percentage(crop_img))
            if black_percentage(crop_img) > 60:
                # try to stitch the next image to see if the image is broken
                if (i+1) != (len(img_list)-1) and black_percentage(self.stitch(crop_img, cv2.imread(img_list[i+1]))) > 90 :
                    crop_img = perspective_transform_and_resize(crop_img)
                    crop_list.append(crop_img)
                    print('Store data and warp')
                    prev = self.stitch_list(img_list[i-self.rewind : (i + 1)], resize = True)
                else:
                    prev = crop_img
            else:
                prev = crop_img
        
        crop_img = perspective_transform_and_resize(crop_img)
        crop_list.append(crop_img)
        #for i, crop in enumerate(crop_list):
        #    cv2.imwrite(f'data/output/out_part_{i}.png', crop)
        assert len(crop_list) <= 2, 'Too long video'
        if len(crop_list) == 2:
            final_img = self.stitch_short(crop_list[0], crop_list[1])
        else:
            final_img = crop_list[0]
        final_img = crop_edge(final_img)
        try:
            final_img = perspective_transform(final_img, get_four_corners(final_img, mode='left_bottom'))
            final_img = perspective_transform(final_img, get_four_corners(final_img, mode='bottom_left'))
        except:
            pass
        final_img = crop_black(final_img)
        cv2.imwrite(os.path.join(self.output_path, self.folder_name+'.png'), final_img)
        
        print(f'Stitching completed for {self.folder_name}')
    
    
    def stitch(self, prev, curr, draw_matches=False):
        sift = cv2.SIFT_create()
        h1, w1 = prev.shape[0:2]
        h2, w2 = curr.shape[0:2]
        prev_crop = prev[:, w1-w2:]
        diff = np.size(prev, axis=1) - np.size(prev_crop, axis = 1)
        kp1, des1 = sift.detectAndCompute(prev_crop, None)
        #kp1, des1 = sift.detectAndCompute(prev, None)
        kp2, des2 = sift.detectAndCompute(curr, None)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        valid_matches = filter_matches(matches, kp1, kp2)
                
        # Extract the coordinates of matching points
        img1_pts = []
        img2_pts = []
        for match in valid_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)
            
        # Draw matches
        if draw_matches:
            matched_image = cv2.drawMatches(prev, kp1, curr, kp2, valid_matches, None, flags=2)
            cv2.imwrite('a.jpg', matched_image)
        
        # Formalize as matrices (for the sake of computing Homography)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img1_pts[:, :, 0] += diff  # Recover its original coordinates
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
        
        # Compute the Homography matrix
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        stitched_img = self.stitch_by_H(prev, curr, H)
        return stitched_img
    
    def stitch_short(self, prev, curr):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(prev, None)
        kp2, des2 = sift.detectAndCompute(curr, None)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        valid_matches = filter_matches(matches, kp1, kp2)
                
        # Extract the coordinates of matching points
        img1_pts = []
        img2_pts = []
        for match in valid_matches:
            img1_pts.append(kp1[match.queryIdx].pt)
            img2_pts.append(kp2[match.trainIdx].pt)
        
        # Formalize as matrices (for the sake of computing Homography)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)
        
        # Compute the Homography matrix
        H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        stitched_img = self.stitch_by_H(prev, curr, H)
        return stitched_img
    
    def stitch_by_H(self, img1, img2, H):
        """Use the key points to stitch the images.
        img1: the image containing frames that have been joint before.
        img2: the newly selected key frame.
        H: Homography matrix, usually from compute_homography(img1, img2).
        """
        # Get heights and widths of input images
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]

        # Store the 4 ends of each original canvas
        img1_canvas_orig = np.float32([[0, 0], [0, h1],
                                       [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        img2_canvas = np.float32([[0, 0], [0, h2],
                                  [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        # The 4 ends of (perspective) transformed img1
        img1_canvas = cv2.perspectiveTransform(img1_canvas_orig, H)

        # Get the coordinate range of output (0.5 is fixed for image completeness)
        output_canvas = np.concatenate((img2_canvas, img1_canvas), axis=0)
        [x_min, y_min] = np.int32(output_canvas.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(output_canvas.max(axis=0).ravel() + 0.5)

        # The output matrix after affine transformation
        transform_array = np.array([[1, 0, -x_min],
                                    [0, 1, -y_min],
                                    [0, 0, 1]])

        # Warp the perspective of img1
        img_output = cv2.warpPerspective(img1, transform_array.dot(H),
                                         (x_max - x_min, y_max - y_min), flags=cv2.INTER_NEAREST)

        for i in range(-y_min, h2 - y_min):
            for j in range(-x_min, w2 - x_min):
                if np.any(img2[i + y_min][j + x_min]):
                    img_output[i][j] = img2[i + y_min][j + x_min]
        return img_output
    
    def stitch_list(self, list_imgs, resize):
        """ stitching list of images, not do perspective transform"""
        assert len(list_imgs) > 0, "too few images"
        
        prev = cv2.imread(list_imgs[0]) if isinstance(list_imgs[0], str) else list_imgs[0]
        for i in range(1, len(list_imgs)):
            curr = cv2.imread(list_imgs[i]) if isinstance(list_imgs[i], str) else list_imgs[i]
            stitched_img = self.stitch(prev, curr)
            crop_img = crop_edge(stitched_img)
            if resize:
                crop_img = imutils.resize(crop_img, height=curr.shape[0])
            prev = crop_img
        return crop_img
    
if __name__ == '__main__':
    vid_list = glob.glob('data/vids/*.MOV')
    for vid in vid_list:
        stitch_clip = StitchingClip(clip_path = vid)
        stitch_clip.extract_frames(rotate= cv2.ROTATE_90_CLOCKWISE)
        stitch_clip.run()
    