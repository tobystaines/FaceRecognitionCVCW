%% extractFeaturesAndTrainModels

% This script takes the data prepared by prepareData.m and uses it to train
% six models:
%       A CNN, using transfer learning based on Alexnet
%       A Random Forest using HOG features
%       A Random Forest using SURF
%       A SVM using HOG features
%       A SVM using SURF
%       A Naive Bayes classifier using HOG features

% All models and extracted feature vectors are saved to file. Some
% variables are cleared from memory after saving, before the completion of
% the script, to avoid memory constraint issues.

%% Create Training, Validation and Test IMDSs

images = imageDatastore('facesFolder/trainingFaces',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');

testImages = imageDatastore('facesFolder/testingFaces',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
save('IMDSs.mat','trainingImages','validationImages','testImages','images')

%% Train a CNN based on Alexnet (Tobynet)
% Pre-process images
% Create image augmenter
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-30,30], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3],...
    'RandYReflection', true);

% Create augmented IMDSs for training, validation and test
% Images are resized and converted to rgb to fit Alexnet input layer
network_input_dims = [227 227];
autrainingImages = augmentedImageDatastore(network_input_dims,trainingImages,...
    'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
auvalidationImages = augmentedImageDatastore(network_input_dims,validationImages,...
    'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
autestImages = augmentedImageDatastore(network_input_dims,testImages,...
    'ColorPreprocessing','gray2rgb'); % Test images are not rotated, reflected, etc.

% Load pre-trained net (Alexnet)
net = alexnet;

% Remove fully connected and classification layers and replace
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImages.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% Set network options
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',6,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',auvalidationImages,...
    'ValidationPatience',2,...
    'ValidationFrequency',numIterationsPerEpoch);

% Train the transfered network
[tobynet, netInfo] = trainNetwork(autrainingImages,layers,options);

% Get training and validation results
trainAcc = netInfo.TrainingAccuracy(end)/100;
valAcc = netInfo.ValidationAccuracy(netInfo.ValidationAccuracy > 0);
valAcc = valAcc(end)/100;

% Test it
truth = testImages.Labels;
prediction = classify(tobynet,autestImages);
testAcc = sum(prediction == truth)/numel(truth);

CNNAcc = [trainAcc valAcc testAcc];

%% Extract HOG Features
% Calculate the size of the HOG feature vector
standard_face_dims = [200 200];
CellSize = [10 10];
BlockSize = [2 2];
BlockOverlap = ceil(BlockSize/2);
NumBins = 9;
BlocksPerImage = floor((standard_face_dims./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1);
N = prod([BlocksPerImage, BlockSize, NumBins]);

% Create a matrix for HOG trainaing, validation and test feauters
trainHOGFeatures = zeros(size(trainingImages.Files,1),N);
valHOGFeatures = zeros(size(validationImages.Files,1),N);
testHOGFeatures = zeros(size(testImages.Files,1),N);

% Extract HOG features from trainaing, validation and test sets
for i = 1:size(trainHOGFeatures,1)
    trainHOGFeatures(i,:) = extractHOGFeatures(readimage(trainingImages,i),'CellSize',CellSize);
end

for i = 1:size(valHOGFeatures,1)
    valHOGFeatures(i,:) = extractHOGFeatures(readimage(validationImages,i),'CellSize',CellSize);
end

for i = 1:size(testHOGFeatures,1)
    testHOGFeatures(i,:) = extractHOGFeatures(readimage(testImages,i),'CellSize',CellSize);
end
% Save the HOG Feature vectors to file
save('HOGFeatures.mat','trainHOGFeatures','valHOGFeatures','testHOGFeatures','-v7.3');

%% Train a HOG SVM
HOGSVM = fitcecoc(trainHOGFeatures, trainingImages.Labels); %, 'OptimizeHyperparameters','auto');
HOGSVM = compact(HOGSVM);

% Test it
trainPrediction = predict(HOGSVM,trainHOGFeatures);
trainAcc = sum(trainPrediction == trainingImages.Labels)/numel(trainingImages.Labels);

valPrediction = predict(HOGSVM,valHOGFeatures);
valAcc = sum(valPrediction == validationImages.Labels)/numel(validationImages.Labels);

testPrediction = predict(HOGSVM,testHOGFeatures);
testAcc = sum(testPrediction == testImages.Labels)/numel(testImages.Labels);

HOGSVMAcc = [trainAcc valAcc testAcc];

% Save labels with CNN model for use processing results in RecogniseFace
tobynet_Labels = cellstr(HOGSVM.ClassNames);
save('tobynet.mat','tobynet','tobynet_Labels');
clear tobynet tobynet_Labels

% Save the model to file
save('HOGSVM.mat','HOGSVM','-v7.3');
clear HOGSVM

%% Train a HOG Random Forest
HOGRandomForestOptRes = zeros(4,2);
HOGRandomForestStore = cell(4,1);
i=1;
for t = [50 100 150 200] % Try forests of 50 - 200 trees
    % Train and compact the model
    HOGRandomForestStore{i} = TreeBagger(t, trainHOGFeatures, trainingImages.Labels);
    HOGRandomForestStore{i} = compact(HOGRandomForestStore{i});
    % Asses performance on training and validation data
    trainPrediction = predict(HOGRandomForestStore{i},trainHOGFeatures);
    trainAcc = sum(trainPrediction == trainingImages.Labels)/numel(trainingImages.Labels);

    valPrediction = predict(HOGRandomForestStore{i},valHOGFeatures);
    valAcc = sum(valPrediction == validationImages.Labels)/numel(validationImages.Labels);
    
    HOGRandomForestOptRes(i,:) = [trainAcc valAcc];
    i = i+1
end
%% 
% After examining results, the 100 tree model is retained 
HOGRandomForest = HOGRandomForestStore{2};
% Test final model
testPrediction = predict(HOGRandomForest,testHOGFeatures);
testAcc = sum(testPrediction == testImages.Labels)/numel(testImages.Labels);
   
HOGRandomForestAcc = [HOGRandomForestOptRes(2,1) HOGRandomForestOptRes(2,2) testAcc];

% Save the model to file
save('HOGRandomForest.mat','HOGRandomForest');
clear HOGRandomForest

%% Train a HOG Naieve Bayes classifier
HOGNB = fitcnb(trainHOGFeatures, trainingImages.Labels);
HOGNB = compact(HOGNB);
% Asses performance on each dataset
trainPrediction = predict(HOGNB,trainHOGFeatures);
trainAcc = sum(trainPrediction == trainingImages.Labels)/numel(trainingImages.Labels);

valPrediction = predict(HOGNB,valHOGFeatures);
valAcc = sum(valPrediction == validationImages.Labels)/numel(validationImages.Labels);

testPrediction = predict(HOGNB,testHOGFeatures);
testAcc = sum(testPrediction == testImages.Labels)/numel(testImages.Labels);

HOGNBAcc = [trainAcc valAcc testAcc];


save('HOGNB.mat','HOGNB');
clear trainHOGFeatures valHOGFeatures testHOGFeatures HOGNB
%% Extract SURF Features
% Create a bag of features
bag = bagOfFeatures(trainingImages);
% Encode features for each dataset, for use with non-SVM models
trainSURFFeatures = encode(bag, trainingImages);
valSURFFeatures = encode(bag, validationImages);
testSURFFeatures = encode(bag, testImages);

% Save the SURF Feature vectors to file
save('SURFFeatures.mat','trainSURFFeatures','valSURFFeatures','testSURFFeatures','-v7.3');
save('bag.mat','bag');
%% Train a SURF SVM

SURFSVM = trainImageCategoryClassifier(trainingImages, bag);

[~,trainTruth,trainPrediction,~] = evaluate(SURFSVM, trainingImages);
trainAcc = sum(trainPrediction == trainTruth)/numel(trainTruth);

[~,valTruth,valPrediction,~] = evaluate(SURFSVM, validationImages);
valAcc = sum(valPrediction == valTruth)/numel(valTruth);

[~,testTruth,testPrediction,~] = evaluate(SURFSVM, testImages);
testAcc = sum(testPrediction == testTruth)/numel(testTruth);

SURFSVMAcc = [trainAcc valAcc testAcc];

% Save the model to file
save('SURFSVM.mat', 'SURFSVM', '-v7.3');

%% Train a SURF Random Forest
% Try forsts of size 50, 100, 150 and 200 trees
SURFRandomForest = cell(4,1);
SURFRandomForestOptRes = zeros(4,2);
i=1;
for t = [50 100 150 200]
    SURFRandomForest{i} = TreeBagger(t, trainSURFFeatures, trainingImages.Labels);
    SURFRandomForest{i} = compact(SURFRandomForest{i});

    trainPrediction = predict(SURFRandomForest{i},trainSURFFeatures);
    trainAcc = sum(trainPrediction == trainingImages.Labels)/numel(trainingImages.Labels);

    valPrediction = predict(SURFRandomForest{i},valSURFFeatures);
    valAcc = sum(valPrediction == validationImages.Labels)/numel(validationImages.Labels);
    SURFRandomForestOptRes(i,:) = [trainAcc valAcc];
    i=i+1
end
%%
% After examining results, the 100 tree model is retained
SURFRandomForest = SURFRandomForest{2};

testPrediction = predict(SURFRandomForest,testSURFFeatures);
testAcc = sum(testPrediction == testImages.Labels)/numel(testImages.Labels);

SURFRandomForestAcc = [SURFRandomForestOptRes(2,1) SURFRandomForestOptRes(2,2) testAcc];

% Save the model to file
save('SURFRandomForest.mat','SURFRandomForest');
clear trainSURFFeatures valSURFFeatures testSURFFeatures SURFRandomForest

%% Collate Results

collatedAcc = num2cell([CNNAcc; HOGSVMAcc; HOGRandomForestAcc; HOGNBAcc; SURFSVMAcc; SURFRandomForestAcc]);
models = {'CNN';'HOG SVM';'HOG Random Forest';...
    'HOG Naieve Bayes';'SURF SVM';'SURF Random Forest'};
collatedAcc = [models collatedAcc];
headers = {'Model','Training','Validation','Test'};
collatedAcc = cell2table(collatedAcc);
collatedAcc.Properties.VariableNames = headers;
save('Acc.mat', 'collatedAcc');