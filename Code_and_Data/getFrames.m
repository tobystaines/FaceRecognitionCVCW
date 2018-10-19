function frameStore = getFrames(vidObj)
%getFrames()
% Takes a videoReader object and returns a struct array containing the 
% individual frames of the video as unint8 images. Adapted from Matlab
% documentation - Read Video Files (full reference in project report).

vidHeight = vidObj.Height;
vidWidth = vidObj.Width;

frameStore = struct('cdata',zeros(vidHeight,vidWidth,3, 'uint8'),... 
                    'colormap',[]);
k = 1;
while hasFrame(vidObj)
    frameStore(k).cdata = readFrame(vidObj);
    k = k+1;
end
end

