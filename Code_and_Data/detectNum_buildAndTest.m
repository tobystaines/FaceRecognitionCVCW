%% Computer Vision Coursework - load data and test detectNum

% This script requires all images and videos from the original file
% provided to have been removed form the subfolder structure and placed in
% a single folder per media type.

%% Load image files and classify using detectNum()

imageStore = loadImages('C:\Users\Toby\Documents\MSc Data Science\3_Computer Vision\CV Coursework\All Images');

imageIDs = zeros(size(imageStore,1),5);
for i = 1:size(imageStore,1)
    num_vec = detectNum(imageStore{i});
    for j = 1:size(num_vec,2)
        imageIDs(i,j) = num_vec(j);
    end
end

% Anywhere that a number is not located in an image a NaN is reutnred.
% These are relabeled as -1 for ease of handling.
imageIDs(isnan(imageIDs)) = -1;

% Any number returned greater than 100 is incorrect. These are relabeled as
% -2. (This should no longer be a problem since implementation of
% multi-number recognition)
imageIDs(imageIDs > 100) = -2;

% Results were also confirmed by visual assesment
imageAcc = length(imageIDs(imageIDs > 0))/length(imageIDs);

%% Load video files and classify using DetectNum()

vidStore = loadVids('C:\Users\Toby\Documents\MSc Data Science\3_Computer Vision\CV Coursework\All Videos');

vidIDs = zeros(size(vidStore,1),1);

for i = 1:size(vidStore,1)
    vidIDs(i) = detectNum(vidStore{i});
end

% Anywhere that a number is not located in a video a NaN is reutnred.
% These are relabeled as -1 for ease of handling.
vidIDs(isnan(vidIDs)) = -1;

% Any number returned greater than 100 is incorrect. These are relabeled as
% -2.
vidIDs(vidIDs > 100) = -2;

% Results were also confirmed by visual assesment
vidAcc = length(vidIDs(vidIDs > 0))/length(vidIDs);

