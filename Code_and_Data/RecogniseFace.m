function [P, I_IDs] = RecogniseFace(I, classifierName, varargin)
% Recognise Face
%
% Inputs:
% 	I: UINT8 image
% 	classifierName: The classifier type selected. 
%       From {'CNN','SVM','RandomForest,'NaiveBayes'}.
% 	featureType: The type of image features to use in classification. 
%       From {'HOG','SURF','NIL'}. Default = NIL.
% 	displayAnnotatedImage: Boolean. Whether or not to display an annotated 
%       version of the original image. Default = true.
%
% Outputs:
% 	P: A matrix representing the people present in an RGB image I.
% 		P is a matrix of size Nx3, where N is the number of people detected in 
% 		the image I.
%
% 		For each person detected, the three columns of P represent:
%       	ID: A unique number associated with each person in training 
%       		database provided.
%       	x: The x location of the person detected in the image (central 
%       		face region)
%       	y: The y location of the person detected in the image (central 
%       		face region)
%
% 	I_IDs: An annotated version of the input image I, with bounding boxes 
% 		around each detected face, labeled with the classification.

%------------- ERROR CHECKING -----------------%
% Check number of var args passed
numvarargs = length(varargin);
if numvarargs > 2
    error('RecogniseFace:TooManyInputs', ...
        'Requires at most 2 optional inputs');
end

% set defaults for optional inputs
optargs = {'NIL',true};

% overwrite defaults with values provided in varargin
optargs(1:numvarargs) = varargin;

% Place optional args in appropriate variables
[featureType, displayAnnotatedImage] = optargs{:};

% Check for valid arguments
validClassifierNames = {'CNN','SVM','RandomForest','NaiveBayes'};
validFeatureTypes = {'HOG','SURF','NIL'};
if ~any(strcmp(validClassifierNames,classifierName))
	error('RecogniseFace:InvalidClassifierName', ...
        'Invalid classifier name - must be one of {"CNN","SVM","RandomForest","NaiveBayes"}');
end
if ~any(strcmp(validFeatureTypes,featureType))
	error('RecogniseFace:InvalidFeatureType', ...
        'Invalid feature type - must be one of {"HOG","SURF","NIL"}');
end
%----------------- MAIN FUNCTION---------------%

% Detect Faces in I, locations stored in BBOX
MinSize = [100 100]; % Adding size thresholds sgnificantly speeds up processing
MaxSize = [250 250];
MergeThreshold = 10;

% Four face detectors are created
FaceDetectorGroup = vision.CascadeObjectDetector('MinSize',[60 60], 'MaxSize',...
    [325 325], 'MergeThreshold', MergeThreshold);
FaceDetectorFFC = vision.CascadeObjectDetector('MinSize',MinSize, 'MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold);
FaceDetectorLBP = vision.CascadeObjectDetector('MinSize', MinSize, 'MaxSize',...
    MaxSize, 'MergeThreshold', MergeThreshold, 'ClassificationModel',...
    'FrontalFaceLBP');
FaceDetectorProfile = vision.CascadeObjectDetector('MinSize', MinSize,...
    'MaxSize', MaxSize, 'MergeThreshold', MergeThreshold,...
    'ClassificationModel', 'ProfileFace');

BBOX = step(FaceDetectorGroup,I);
if isempty(BBOX) % If the FFC detector finds no face, assume not a group image
    I = imresize(I,[1024 768]); % downsize
    BBOX = step(FaceDetectorFFC,I);
    if isempty(BBOX) % If the FFC detector finds no face, try the LBP detector
        BBOX = step(FaceDetectorLBP,I);
        if isempty(BBOX) % If the LBP detector finds no face, try the profile detector
            BBOX = step(FaceDetectorProfile,I);
        end
    end
end

% Create a gallery of extracted faces
standard_face_dims = [200 200];
faceGallery = cell(size(BBOX,1),1);
for face = 1:size(BBOX,1)
	loc = BBOX(face,:);
	box = imcrop(I,loc);
	box = imresize(box, standard_face_dims);
	faceGallery{face} = rgb2gray(box);
    %file_name = strcat(num2str(face), '.jpg');
    %imwrite(faceGallery{face}, file_name);
    %faceGallery{face} = imread(file_name);
end

% Create the empty matrix P, with one row per detected face, then fill in columns x and y
P = zeros([size(BBOX,1),3]);
for face = 1:size(BBOX,1)
    % X and Y coordinates of face centres calculated using BBOX values.
    % Round is used to account for face regions where width/height is an
    % odd number of pixels.
    P(face,2) = round(BBOX(face,1) + (BBOX(face,3)/2));
    P(face,3) = round(BBOX(face,2) + (BBOX(face,4)/2));
end

% Pass detected faces to selected classifier
if strcmp(classifierName,'CNN')
	if strcmp(featureType,'NIL')
        load('tobynet.mat','tobynet','tobynet_Labels');
        for face = 1:size(faceGallery,1)
            input = imresize(faceGallery{face},[227 227]);
            input = input(:,:,[1 1 1]);
            % predict face ID using CNN
            [labelIdx,~] = classify(tobynet,input);
            P(face,1)= str2double(tobynet_Labels{labelIdx});
        end
    else
        error('RecogniseFace:InvalidCNNFeatureType', ...
			'Invalid classifierName/featureType combination - CNN expects featureType "NIL"');
    end
elseif strcmp(classifierName,'SVM')
	if strcmp(featureType,'HOG')
		load('HOGSVM.mat','HOGSVM')
        HOGFeatures = zeros(size(faceGallery,1),12996);
		for face = 1:size(faceGallery,1)
            % Extract HOG features from image
            HOGFeatures(face,:) = extractHOGFeatures(faceGallery{face},'CellSize',[10 10]);
        end
        % predict face ID using HOG SVM
        labels = char(predict(HOGSVM,HOGFeatures));
        for i = 1:size(labels,1)
            P(i,1) = str2double(labels(i,1:3));
        end
	elseif strcmp(featureType,'SURF')
		load('SURFSVM.mat','SURFSVM')
		for face = 1:size(BBOX,1)
			% predict face ID using SURF SVM
			class = predict(SURFSVM,faceGallery{face});
            P(face,1) = str2double(SURFSVM.Labels{class});
		end
    else
        error('RecogniseFace:InvalidSVMFeatureType', ...
			'Invalid classifierName/featureType combination - SVM expects featureType in {"HOG","SURF"}');
	end
elseif strcmp(classifierName,'RandomForest')
	if strcmp(featureType,'HOG')
        load('HOGRandomForest.mat','HOGRandomForest');
        HOGFeatures = zeros(size(faceGallery,1),12996);
		for face = 1:size(faceGallery,1)
            % Extract HOG features
            HOGFeatures(face,:) = extractHOGFeatures(faceGallery{face},'CellSize',[10 10]);
        end
        % predict face ID using HOG RF
        labels = char(predict(HOGRandomForest,HOGFeatures));
        for i = 1:size(labels,1)
            P(i,1) = str2double(labels(i,1:3));
        end
	elseif strcmp(featureType,'SURF')
        load('SURFRandomForest.mat','SURFRandomForest');
        load('bag.mat','bag');
        SURFFeatures = zeros(size(BBOX,1),500);
		for face = 1:size(BBOX,1)
            % Extract SURF features
			SURFFeatures(face,:) = encode(bag,faceGallery{face});
        end
        % predict face ID using SURF RF
        labels = char(predict(SURFRandomForest,SURFFeatures));
        for i = 1:size(labels,1)
            P(i,1) = str2double(labels(i,1:3));
        end
    else
        error('RecogniseFace:InvalidRFFeatureType', ...
			'Invalid classifierName/featureType combination - RandomForest expects featureType in {"HOG","SURF"}');
    end
elseif strcmp(classifierName,'NaiveBayes')
    if strcmp(featureType,'HOG')
        load('HOGNB.mat','HOGNB');
        HOGFeatures = zeros(size(faceGallery,1),12996);
		for face = 1:size(faceGallery,1)
            % Extract HOG features
            HOGFeatures(face,:) = extractHOGFeatures(faceGallery{face},'CellSize',[10 10]);
        end
        % predict face ID using HOG NB
        labels = char(predict(HOGNB,HOGFeatures));
        for i = 1:size(labels,1)
            P(i,1) = str2double(labels(i,1:3));
        end
    else
        error('RecogniseFace:InvalidNBFeatureType', ...
			'Invalid classifierName/featureType combination - NaiveBayes expects featureType "HOG"');    
    end
end

% Annotate original image with bounding boxes and IDs added
P(isnan(P(:,1))) = 0;
if size(BBOX,1)>0
    I_IDs = insertObjectAnnotation(I,'rectangle',BBOX,P(:,1),'FontSize',42);
else
    I_IDs = I;
end
if displayAnnotatedImage == true
    figure;
	imshow(I_IDs);
end
end

