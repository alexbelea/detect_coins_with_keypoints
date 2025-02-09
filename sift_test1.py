import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def SIFT_compare():
    print('teh SIFT function is runnin')
    root = os.getcwd()   # saves the current working folder to a string to use later _FANCY_

    # Load Image 1
    imgPath1 = os.path.join(root, '2022-dame-vera-lynn.jpg')
    imgGray_1 = cv.imread(imgPath1, cv.IMREAD_GRAYSCALE)
    if imgGray_1 is None:
        print(f"Error: Could not load image at {imgPath1}")
        return

    # Load Image 2
    imgPath2 = os.path.join(root, '2022-dame-vera-lynn_skew.jpg')
    # imgPath2 = os.path.join(root, 'u2_great-fire-of-london_coin_thumb.jpg')
    imgGray_2 = cv.imread(imgPath2, cv.IMREAD_GRAYSCALE)
    if imgGray_1 is None:
        print(f"Error: Could not load image 2 at {imgPath2}")
        return

    # Create SIFT object
    sift = cv.SIFT_create()

    # Detect Keypoints and compute descriptors image 1
    keypoints_1, descriptors_1 = sift.detectAndCompute(imgGray_1, None)
    imgGray_1 = cv.drawKeypoints(imgGray_1, keypoints_1, imgGray_1,
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Keypoints image 2

    keypoints_2, descriptors_2 = sift.detectAndCompute(imgGray_2, None)
    imgGray_2 = cv.drawKeypoints(imgGray_2, keypoints_2, imgGray_2,
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Descriptor Matching
    # Initialize a Brute-Force Matcher (you can also use other matchers like FLANN)
    bf = cv.BFMatcher_create(cv.NORM_L2) 

    # Perform matching: Find best matches between descriptors_1 and descriptors_2
    matches = bf.match(descriptors_1, descriptors_2)

    # Sort matches by distance (the lower the distance, the better the match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Distance Thresholding
    good_matches_distance = []
    distance_threshold = 50  # Adjust this value based on experimentation


    for match in matches:
        print('Here\'s the distances:')
        print(match.distance)
        if match.distance < distance_threshold:
            good_matches_distance.append(match)
    
    print(f"Matches after distance threshold ({distance_threshold}): {len(good_matches_distance)}")

    # Draw only the top 'numMatches' good matches (e.g., top 10)
    numMatches = 10 # You can adjust this number
    imgMatches = cv.drawMatches(imgGray_1, keypoints_1, imgGray_2, keypoints_2,
                                   matches[:numMatches], None,
                                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    img_matches_distance_thresh = cv.drawMatches(imgGray_1, keypoints_1, imgGray_2, keypoints_2,
                                    good_matches_distance[:numMatches], None, # Draw top 'numMatches' of good matches
                                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Display the images and the matches
    plt.figure(figsize=(15, 5)) # Adjust figure size for better visualization

    plt.subplot(1, 3, 1) # 1 row, 3 columns, plot in the 1st position
    plt.imshow(imgGray_1, cmap='gray')
    plt.title('Image 1 Keypoints')
    plt.xticks([]), plt.yticks([]) # Turn off axis ticks

    plt.subplot(1, 3, 2) # 1 row, 3 columns, plot in the 2nd position
    plt.imshow(imgGray_2, cmap='gray')
    plt.title('Image 2 Keypoints')
    plt.xticks([]), plt.yticks([]) # Turn off axis ticks

    plt.subplot(1, 3, 3) # 1 row, 3 columns, plot in the 3rd position
    plt.imshow(imgMatches)
    plt.title(f'Top {numMatches} Matches')
    plt.xticks([]), plt.yticks([]) # Turn off axis ticks


    plt.tight_layout() # Adjust subplot parameters for a tight layout
    plt.show()
  
    '''  # Plot Image 1
    plt.figure()
    plt.imshow(imgGray_1)
    plt.show()


    # Plot Image 2
    plt.figure()
    plt.imshow(imgGray_2)
    plt.show()'''


if __name__ == "__main__":
    SIFT_compare()
