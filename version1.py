'''
SOURCES:
https://www.geeksforgeeks.org/image-stitching-with-opencv/
https://youtube.com/watch?v=v9JARVu74CI&t=14s
https://github.com/OpenStitching/stitching
https://dronemapper.com/sample_data/
https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html#a02ef4d594b33d091767cbfe442aefb8a

'''

import cv2
import numpy as np
import os

'''
@params
folder: must be a folder name
debug: 0 = no debug; 1 = light debug; 2 = detailed debug
'''
def stitch(folder, debug = 0):
    print("\n----------------------\n")
    print("Run Start")
    images = []
    if debug > 0: print("Scanning Images in folder:",folder)
    for filename in os.listdir(folder):
        #Create full file path 
        f = os.path.join(folder, filename)
        # Skipping if it is a file
        if not os.path.isfile(f):
            continue
        #Save file in images
        images.append(cv2.imread(f))
        if debug == 2: print(f, "Loaded")
    if debug > 0: print("Done Scanning")

    #Finds features and adds details related
    #Uses SIFT algorithm (Scale-Invariant Feature Transform)
    #Also could try cv2.ORB_create()
    sift = cv2.SIFT_create()
    if debug == 2: print("Sift class created")
    #Storing keypoints and descriptors here
    keypoints = []
    descriptors = []
    
    if debug > 0: print("Loading keypoints and descriptors. Please wait...")
    imagecounter = 0
    for image in images:
        #Magic openCV detection
        #NOTE look into why the tutorial recommends None as the second param
        key, desc = sift.detectAndCompute(image, None)
        keypoints.append(key)
        descriptors.append(desc)
        if debug == 2: print("Stored Keypoints and Descriptors for Image: ", imagecounter)
        imagecounter += 1

    if debug > 0: print("Keypoint and Descriptors loaded")

    #whether to put create or not is debated between sites
    #L1 or L2 is good with SIFT
    #HAMMING is good with ORB or BRISK (i.e. NORM_HAMMING)
    '''
    If it is false, this is will be default BFMatcher behaviour
    when it finds the k nearest neighbors for each query descriptor.

    If crossCheck==true, then the knnMatch() method with k=1 will
    only return pairs (i,j) such that for i-th query descriptor the
    j-th descriptor in the matcher's collection is the nearest and
    vice versa, i.e. the BFMatcher will only return consistent pairs.

    Such technique usually produces best results with minimal number
    of outliers when there are enough matches. This is alternative to
    the ratio test, used by D. Lowe in SIFT paper.
    
    No I (Jasper) didn't write that. I dont got that many brain cells
    left. Copied from the 5th link
    '''
    #if debug > 0: print("Finding matches...")
    
    #bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
    #matches = bf.match(descriptors[0],descriptors[1])
    
    #if debug > 0: print("Matches Found")

    print("Stitching...")
    stitcher = cv2.SIFT_create(0)
    status, output = stitcher.stitch(images)

    possibleStatus = [
        "OK",
        "ERR_NEED_MORE_IMGS",
        "ERR_HOMOGRAPHY_EST_FAIL",
        "ERR_CAMERA_PARAMS_ADJUST_FAIL"
    ]

    if(status == 0):
        cv2.imshow('1',output) 
        cv2.waitKey(0)
    else:
        print("Error: ", possibleStatus[status])

    print("Run End")





stitch('C:\\Users\jaspe\Documents\Github_Local\mapping-tjuav\ImgSampleC', debug = 2)