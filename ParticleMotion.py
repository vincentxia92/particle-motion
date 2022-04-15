import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import cv2
from skimage.transform import resize, rescale
import glob


def motion_detector_gist():
  
  frames = glob.glob("Sample_29/2022.02.06_11.12.34_CAM0/*.png")#2022.02.06_11.12.34_CAM0

  previous_frame = None

  for frame_addr in frames:

    # 1. Load image; convert to RGB
    #img_brg = np.array(ImageGrab.grab())
    #img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
	
    # 2. Prepare image; grayscale and blur
    #prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	
    prepared_frame = io.imread(frame_addr)
	
   
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

    # 2. Calculate the difference
    if (previous_frame is None):
      # First frame; there is no previous one yet
      previous_frame = prepared_frame
      continue

    # 3. Set previous frame and continue if there is None
    if (previous_frame is None):
      # First frame; there is no previous one yet
      previous_frame = prepared_frame
      continue

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    previous_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]

    # 6. Find and optionally draw contours
    _, contours,_  = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Comment below to stop drawing contours
    cv2.drawContours(image=prepared_frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # Uncomment 6 lines below to stop drawing rectangles
    # for contour in contours:
      # if cv2.contourArea(contour) > 500:# too large: skip!
          # continue
      # (x, y, w, h) = cv2.boundingRect(contour)
      # cv2.rectangle(img=prepared_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    #cv2.imshow('Motion detector', prepared_frame)
    cv2.imshow('Motion detector',rescale(prepared_frame,0.25,anti_aliasing=False))
    cv2.waitKey(0)

    if (cv2.waitKey(30) == 27):
      # out.release()
      break
 
  # Cleanup
  cv2.destroyAllWindows()
  
  
if __name__ == "__main__":
  motion_detector_gist()
