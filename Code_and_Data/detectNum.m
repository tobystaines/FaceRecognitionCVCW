function num_vec = detectNum(inputMedia, varargin)
% Detect Num
%
% Detects a numeric character or characters in an image or video file and 
% returns those characters as an integer, or vector of integers if multiple
% separate numbers are detected.

% Check number of var args passed
numvarargs = length(varargin);
if numvarargs > 1
    error('detectNum:TooManyInputs', ...
        'Requires at most 1 optional inputs');
end
% set default for optional singleNumOnly input
singleNumOnly = {false};
% overwrite default with value provided in varargin
singleNumOnly(1:numvarargs) = varargin;
% Convert from cell array to logical
singleNumOnly = singleNumOnly{1};

% Check if object passed is a filepath for a media object or an already 
% loaded media object
if isa(inputMedia, 'char')
    if endsWith(upper(inputMedia), '.JPG')
        inputMedia = imread(inputMedia);
    elseif endsWith(upper(inputMedia), '.MOV')
        inputMedia = VideoReader(inputMedia);
    else
        error('detectNum:InvalidInputType', ...
        'Input must be uint8, VideoReader or a valid pathname ending .jpg or .mov');
    end
end

if isa(inputMedia, 'uint8')
    I = inputMedia;
    
    % Ensure image is a standard size
    image_dim = [1024 768];
    I = imresize(I, image_dim);

    % Blur - makes things worse
    %f = ones(3,3)/9;
    %I = imfilter(I, f);

    % Convert to grayscale then binary
    I = rgb2gray(I);
    I = imcomplement(imbinarize(I));

    % Initialize the blob analysis System object(TM)
    blobAnalyzer = vision.BlobAnalysis('MaximumCount', 500);

    % Run the blob analyzer to find connected components and their statistics.
    [area, ~, roi] = step(blobAnalyzer, I);
    
    % Keep regions that meet the area constraint
    areaConstraint = (area > 30 & area < 1100);
    roi = double(roi(areaConstraint, :));

    % Compute the aspect ratio.
    width  = roi(:,3);
    height = roi(:,4);
    aspectRatio = width ./ height;

    % An aspect ratio between 0.25 and 1 is typical for individual characters
    % as they are usually not very short and wide or very tall and skinny.
    roi = roi( aspectRatio > 0.25 & aspectRatio < 1 ,:);

    % Remove regions close to the edge of the image
    roi = roi(roi(:,1) > 200 & ...
              roi(:,1) + roi(:,3) < size(I,2) - 200 & ...
              roi(:,2) > 350 & ...
              roi(:,2) + roi(:,4) < size(I,1) - 100,:);
          
    % Filter out rois which are not overlapping with, or close to, another roi
    roi = mergeOverlappingRois(roi);



    % Run OCR on remaining rois, looking for only numerical characters
    results = ocr(I, roi, 'CharacterSet', '0123456789', 'TextLayout','Word');

    % Remove results where no numbers were detected
    for r = size(results,1):-1:1
       if isnan(str2double(results(r,1).Text))
           results(r,:) = [];
       end
    end

    % Remove low confidence results
    results = results([results.WordConfidences] > 0.3);
    
    % If only a single number has been requested, take the one with highest
    % confidence
    if (singleNumOnly == true && size(results,1) > 0)
        results = results([results.WordConfidences] == max(results.WordConfidences));
    end
    % Take results and put characters together into a single integer output

    if size(results,1) > 0
        num_vec = zeros(1,size(results,1));
        for r = 1:size(results,1)
            num = strrep(results(r,1).Text," ","");
            num_vec(1,r) = str2double(num);
        end
    else
        % If no characters have been detected return NaN
        num_vec = 0/0;
    end
        
elseif isa(inputMedia, 'VideoReader')
    % Get the individual frames and run detectNum() on each frame, keeping 
    % the results in a vector
    frames = getFrames(inputMedia);
    num_vec = zeros(size(frames,2),1);
    for f = 1:size(frames,2)
        num_vec(f) = detectNum(frames(1,f).cdata, true);
    end
    % Find the most commonly detected number and return this
    num_vec = mode(num_vec);
end
end

