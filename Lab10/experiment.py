import cv2
from train import processFiles, trainSVM
from detector import Detector
import numpy as np

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """
    # TODO: You need to adjust hyperparameters
    # Extract HOG features from images in the sample directories and 
    # return results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,hog_features=True)


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)


    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap)


def experiment2():
    # feature_data = processFiles(pos_dir, neg_dir, recurse=True, channels=[0, 1, 2], 
    #                             output_file=True, output_filename="feature_data.p",
    #                             hog_features=True, hist_features=True, spatial_features=True, 
    #                             hog_lib="cv", size=(64, 64), hog_bins=16, pix_per_cell=(8, 8), 
    #                             cells_per_block=(2, 2),
    #                             hist_bins=16, spatial_size=(20, 20))
    
    feature_data = np.load("feature_data.p", allow_pickle=True)
    
    # classifier_data = trainSVM(feature_data=feature_data, C=1000, output_file=True, output_filename="classifier_data.pt")
    classifier_data = np.load("classifier_data.pt", allow_pickle=True)

    

    detector = Detector(
    init_size=(160, 160), x_range=(0.2, 1), y_range=(0.5, 0.9), x_overlap=0.5, y_step=0.025
    ).loadClassifier(classifier_data=classifier_data)
  
    cap = cv2.VideoCapture(video_file)

    detector.detectVideo(video_capture=cap, threshold=30, min_bbox=(30, 30))


if __name__ == "__main__":
    # experiment1()
    experiment2() 


