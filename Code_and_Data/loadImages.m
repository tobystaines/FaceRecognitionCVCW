function store = loadImages(filePath)
% loadImages
% 
% Reads through the contents of a folder and loads all JPEG files in to
% a cell array

srcFiles = dir(strcat(filePath,'\*.jpg'));
store = cell(size(srcFiles,1),1);
image_dim = [1024 768];

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
        store{i} = imresize(I,image_dim);
    end
        
end


