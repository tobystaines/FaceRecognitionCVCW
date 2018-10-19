function vidStore = loadVids(filePath)
% loadVids
% 
% Reads through the contents of a folder and loads all MOV files in to
% a cell array

srcFiles = dir(strcat(filePath,'\*.mov'));
vidStore = cell(size(srcFiles,1),1);


for i = 1:size(srcFiles,1)
    if srcFiles(i).name ~= "."  && srcFiles(i).name ~= ".." 
        filename = fullfile(srcFiles(i).folder,srcFiles(i).name);
        
        % Create a videoReader object to store the video
        vidStore{i} = VideoReader(filename);

    end
        
end

