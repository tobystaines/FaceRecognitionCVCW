# Computer Vision Coursework
Author: Toby Staines

### Files included in submission:

-Main Report: CV_Project Report_TS_v1.0.pdf
-ConfusionMatrix.xlsx: A confusion matrix was requested in the specification, but a 54 class confusion matrix is too large to fit sensibly in a word document, so is included here as an appendix.

#### Core Functions:
- detectNum(): As described in the project report
- RecogniseFace():As described in the project report


#### Supporting Functions:
- loadImages(): Takes a given filepath and reads all .JPG files contained in that folder, checks their orientation is correct and then saves them in a cell vector, which is returned.
- loadVids():   Takes a given file path and reads all .MOV files contained in that folder in to VideoReader objects, and saves these in a cell array, which is returned.
- getFrames():  Takes a VideoReader object and reads the video, saving each frame in a struct array, which is then returned
- mergeOverlappingRois():Takes a regions of interest matrix and merges regions which are overlapping, or close to, other regions. Non-overlapping ROIs are removed.

#### Scripts:
- prepareData.m - This script prepares the raw data. It runs through the folder containing the raw data, replicates the subfolder structure within a new folder called 'vidFrames', loads video files, splits them into separate images, and saves the images into the appropriate new subfolder. It then replicates the subfolder structure again and runs all images (video frames and original still images) through a face detector and saves standardised greyscale face images in appropriate folders. Requires data to be in a folder call 'Images', containing one subfolder for each class with all of the data for that class.

- detectNum_buildAndTest.m - This script was used in the development of the detectNum() function. It loads the still images, passes them through the detect num function and asses the results. It then does the same for the video files (passing them to the function as videoReader objects). Requires all images and videos from the original file provided to have been removed from the subfolder structure and placed in a single folder per media type ('All Videos' and 'All Images'). The file path will need to be changed.

- ExtractFeaturesAndTrainModels.m - This script replicates the training and testing of the six classifiers included in the final RecogniseFace function.

### Files Not Included:
The following data files contain the models required by RecogniseFace. They are not uploaded to GitHub due to size constraints:
- tobynet.mat         
- HOGNB.mat    
- HOGRandomForest.mat 
- HOGSVM.mat
- SURFRandomForest.mat  
- SURFSVM.mat
- bag.mat - bag of words used by SURFRandomForest           
