%% Get Individual Frames as Images from Videos

% Create directory for video frames - vidFrames
% For each folder in 'Images':
%   Create a folder in vidFrames with the same name
%   Load all videos
%   For each video in folder:
%       getFrames
%       Save the frames in to vidFrames subfolder

addpath(cd);
folder_name = 'Images';
mkdir vidFrames
D = dir(folder_name);
D = D(~ismember({D.name}, {'.', '..'}));
D = D(find(vertcat(D.isdir))); % Keep only names of folders
for i = 1:numel(D)
    currentD = D(i).name; % Get the current subdirectory name
    new_folder = fullfile('vidFrames\', currentD);
    mkdir(new_folder);
    vidStore = loadVids(fullfile(folder_name, currentD));
    cd(new_folder);
    for vid = 1:size(vidStore,1)
        frames = getFrames(vidStore{vid});        
        for f = 1:size(frames,2)
            file_name = strcat(num2str(vid), '_', num2str(f), '.jpg');
            imwrite(frames(f).cdata, file_name );                           
        end        
    end
    cd ../../
end

%% Detect Faces in the vidImage Set and Save to File

% Create new directory 'facesFolder'
% for each subfolder:
%   create a new subfolder in facesFolder
%   load all images
%   for each image loaded:
%       run face detector
%       for each face:
%           save to the new folder


% Detect faces in images and store in grayscale in a cell array called 
% 'faceGallery'.
MinSize = [100 100]; % Adding size thresholds sgnificantly speeds up processing
MaxSize = [250 250];
MergeThreshold = 10;

FaceDetectorFFC = vision.CascadeObjectDetector('MinSize',MinSize, 'MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold);
FaceDetectorLBP = vision.CascadeObjectDetector('MinSize', MinSize, 'MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold, 'ClassificationModel',...
    'FrontalFaceLBP');
FaceDetectorProfile = vision.CascadeObjectDetector('MinSize', MinSize,...
    'MaxSize', MaxSize, 'MergeThreshold', MergeThreshold,...
    'ClassificationModel', 'ProfileFace');


standard_face_dims = [200, 200]; % All faces will be rescaled to a standard size

addpath(cd);
folder_name = 'vidFrames';
mkdir facesFolder
D = dir(folder_name);
D = D(~ismember({D.name}, {'.', '..'}));
D = D(find(vertcat(D.isdir))); % Keep only names of folders
for i = 1:numel(D)
    currentD = D(i).name; % Get the current subdirectory name
    new_folder = fullfile('facesFolder\', currentD);
    mkdir(new_folder);
    data = loadImages(fullfile(folder_name, currentD));
    faceGallery = cell(size(data,1),5);
    cd(new_folder);
        for j = 1:size(faceGallery,1)
            bbox = step(FaceDetectorFFC,data{j});
            if isempty(bbox)
                bbox = step(FaceDetectorLBP,data{j});
                if isempty(bbox)
                    bbox = step(FaceDetectorProfile,data{j});
                end
            end
            for face = 1:size(bbox,1)
                loc = bbox(face,:);
                box = imcrop(data{j},loc);
                box = imresize(box, standard_face_dims);
                faceGallery{j, face} = rgb2gray(box);
                file_name = strcat(num2str(i), '_', num2str(j), '_', num2str(face), '.jpg');
                imwrite(faceGallery{j, face}, file_name );
            end
            bbox = [];
        end
    faceGallery = {};
    cd ../../
end

%% Detect Faces in the Still Image Set and Save to File

% for each subfolder:
%   load all images
%   for each image loaded:
%       run face detector
%       for each face:
%           save to the new folder
folder_name = 'Images';

D = dir(folder_name);
D = D(~ismember({D.name}, {'.', '..'}));
D = D(find(vertcat(D.isdir))); % Keep only names of folders
for i = 1:numel(D)
    currentD = D(i).name; % Get the current subdirectory name
    new_folder = fullfile('facesFolder\', currentD);
    data = loadImages(fullfile(folder_name, currentD));
    faceGallery = cell(size(data,1),5);
    cd(new_folder);
    for j = 1:size(faceGallery,1)
        bbox = step(FaceDetectorFFC,data{j});
        if isempty(bbox)
            bbox = step(FaceDetectorLBP,data{j});
            if isempty(bbox)
                bbox = step(FaceDetectorProfile,data{j});
            end
        end
        for face = 1:size(bbox,1)
            loc = bbox(face,:);
            box = imcrop(data{j},loc);
            box = imresize(box, standard_face_dims);
            faceGallery{j, face} = rgb2gray(box);
            file_name = strcat('img_', num2str(i), '_', num2str(j), '_', num2str(face), '.jpg');
            imwrite(faceGallery{j, face}, file_name );
        end
        bbox = [];
    end
    faceGallery = {};
    cd ../../
end

%% Split Faces into Test and Train

faces = imageSet('facesFolder','recursive');
[training,test] = partition(faces,[0.8 0.2]);

% For each of [training test]:
%   create a new folder
%   copy the facesFolder file structure
%   for each imageSet:
%       Write faces in to the new sub-folder with the name
%       imageSet.Description


folders = {training 'trainingFaces';...
           test 'testingFaces'};
cd facesFolder
for i = 1:2
   mkdir(folders{i,2});
   for j = 1:numel(D)
       currentD = D(j).name; 
       new_folder = fullfile(folders{i,2}, currentD);
       mkdir(new_folder)
       for k = 1:folders{i,1}(j).Count
           file_name = strcat(num2str(j),'_',num2str(k),'.jpg');
           imwrite(read(folders{i,1}(j),k),fullfile(new_folder,file_name));
       end
   end
   
end
cd ../
%% Get faces from group pictures
%% First, load the group images
srcFiles = dir('Images\Group\*.jpg');
store = cell(size(srcFiles,1),1);
%image_dim = [1024 768];

for i = 1:size(srcFiles,1)
    if srcFiles(i).name ~= "."  && srcFiles(i).name ~= ".." 
        filename = fullfile(srcFiles(i).folder,srcFiles(i).name);
        I = imread(filename);
        % Check whether file has been read in with the correct orientation
        % (as defined by the thumbnail view in Windows Explorer) and if
        % not, correct it. This if statement is taken from MATLAB Answers
        % (full reference in project report).
        info = imfinfo(filename);
        if isfield(info,'Orientation')
           orient = info(1).Orientation;
           switch orient
             case 1
                %normal, leave the data alone
             case 2
                I = I(:,end:-1:1,:);         %right to left
             case 3
                I = I(end:-1:1,end:-1:1,:);  %180 degree rotation
             case 4
                I = I(end:-1:1,:,:);         %bottom to top
             case 5
                I = permute(I, [2 1 3]);     %counterclockwise and upside down
             case 6
                I = rot90(I,3);              %undo 90 degree by rotating 270
             case 7
                I = rot90(I(end:-1:1,:,:));  %undo counterclockwise and left/right
             case 8
                I = rot90(I);                %undo 270 rotation by rotating 90
             otherwise
                warning('unknown orientation %g ignored\n', orient);
           end
         end
        store{i} = I;
    end
end
%% Divide group photo set in to train/test

[groupTrainIdx, groupTestIdx, ~] = dividerand(size(store,1),0.8,0.2);
groupTrain = store(groupTrainIdx,:);
groupTest = store(groupTestIdx,:);
%% Save the train and test images to separate folders
cd Images
mkdir Group_Train
mkdir Group_Test

cd Group_Train
for i = 1:size(groupTrain,1)
    file_name = strcat(num2str(i),'.jpg');
    imwrite(groupTrain{i},file_name);
end
cd ../
cd Group_Test
for i = 1:size(groupTest,1)
    file_name = strcat(num2str(i),'.jpg');
    imwrite(groupTest{i},file_name);
end
cd ../../
cd facesFolder
% Create new folders for the faces extracted from the group images
cd trainingFaces
mkdir Group_Train
cd ../
cd testingFaces
mkdir Group_Test
cd ../../
%% Detect faces in group images, standardise size and save to file

% MinSize = [100 100]; Detectors redefined without min size, in order to
%                      detect faces at the back of the room
MaxSize = [250 250];
MergeThreshold = 10;

FaceDetectorFFC = vision.CascadeObjectDetector('MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold);
FaceDetectorLBP = vision.CascadeObjectDetector('MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold, 'ClassificationModel',...
    'FrontalFaceLBP');
FaceDetectorProfile = vision.CascadeObjectDetector(...
    'MaxSize', MaxSize, 'MergeThreshold', MergeThreshold,...
    'ClassificationModel', 'ProfileFace');

list = {groupTrain,groupTest};
folder_names = {'trainingFaces/Group_Train','testingFaces/Group_Test'};

for j = 1:2
    faceGallery = cell(size(list{j},1),100);
    for i = 1:size(list{j},1)
        bbox = step(FaceDetectorFFC,list{j}{i});
        bbox = [bbox;step(FaceDetectorLBP,list{j}{i})];
        bbox = [bbox;step(FaceDetectorProfile,list{j}{i})];
        for face = 1:size(bbox,1)
            loc = bbox(face,:);
            box = imcrop(list{j}{i},loc);
            box = imresize(box, standard_face_dims);
            faceGallery{i, face} = rgb2gray(box);
            file_name = strcat(num2str(i), '_', num2str(face), '.jpg');
            imwrite(faceGallery{i, face}, fullfile('facesFolder',folder_names{j},file_name));
        end
    end
    faceGallery = {};
end
%% Extract HoG Features from group images in preparation for clustering

faceGallery = imageSet('facesFolder/Group_Test');
% Set HOG parameters
CellSize = [8 8];
BlockSize = [2 2];
BlockOverlap = ceil(BlockSize/2);
NumBins = 9;

BlocksPerImage = floor((standard_face_dims./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1);
N = prod([BlocksPerImage, BlockSize, NumBins]);

% Create a matrix for HOG feautes
hogFeatures = zeros(faceGallery.Count,N);

% Extract HOG features
for i = 1:size(hogFeatures,1)
    hogFeatures(i,:) = extractHOGFeatures(read(faceGallery,i));
end

%% K-Means on HOG (k=60)

grouping = kmeans(hogFeatures,60);
% Create a new folder and save the clustered images to it
mkdir clusters_hog
cd clusters_hog
for i = 1:faceGallery.Count
    file_name = strcat(num2str(grouping(i)),'_',num2str(i),'.jpg');
    imwrite(read(faceGallery,i),file_name);
end
cd ../

%% Exract Local Binary Pattern Features

lbpFeatures = zeros(faceGallery.Count,1000);
CellSize = [20 20];
for i = 1:faceGallery.Count
    lbpFeatures(i,:) = extractLBPFeatures(read(faceGallery,i),'CellSize',CellSize,'Upright',false);
end
%% K-Means on LBP (k=58)

grouping = kmeans(lbpFeatures,58);
% Create a new folder and save the clustered images to it
mkdir clusters_LBP
cd clusters_LBP
for i = 1:faceGallery.Count
    file_name = strcat(num2str(grouping(i)),'_',num2str(i),'.jpg');
    imwrite(read(faceGallery,i),file_name);
end
cd ../

