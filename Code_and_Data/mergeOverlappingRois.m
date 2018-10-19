function mergedRoi = mergeOverlappingRois(roi)
%mergeOverlappingRois 
%   Takes a regions of interest matrix and merges regions which
%   are overlapping, or close to, other regions. Non-overlapping ROIs are
%   removed.

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
xmin = roi(:,1);
ymin = roi(:,2);
xmax = xmin + roi(:,3) - 1;
ymax = ymin + roi(:,4) - 1;

% Expand the bounding boxes by a small amount.
expansionAmount = 0.03;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;

% Compute the overlap ratio
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1);
overlapRatio(1:n+1:n^2) = 0;

% Create the graph
g = graph(overlapRatio);

% Find the connected text regions within the graph
componentIndices = conncomp(g);

% Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

% Compose the merged bounding boxes using the [x y width height] format.
mergedRoi = [xmin ymin xmax-xmin+1 ymax-ymin+1];

% Remove bounding boxes that only contain one text region
numRegionsInGroup = histcounts(componentIndices);
mergedRoi(numRegionsInGroup == 1, :) = [];

end

