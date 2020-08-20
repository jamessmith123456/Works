function feature = feature_densityfeature(rgbImg, binImg)

[R, C, channels] = size(rgbImg);

if islogical(binImg)
    thres = 0;
else
    thres = 128;
end

% centr = regionprops(binImg, 'centroid');
% centx = int32(centr(end).Centroid(1));
% centy = int32(centr(end).Centroid(2));

boundingbox = regionprops(binImg, 'BoundingBox');
bin_top = round(boundingbox(end).BoundingBox(2));
bin_left = round(boundingbox(end).BoundingBox(1));
bin_height = boundingbox(end).BoundingBox(4);
bin_width = boundingbox(end).BoundingBox(3);
%bin_bot = bin_top + bin_height - 1;
%bin_right = bin_left + bin_width - 1;

rhist = zeros(bin_width,channels);
for i = 1:bin_width
    for j = 1:channels
        temphist = rgbImg(binImg(:,bin_left+i-1)>thres, bin_left+i-1, j);
        rhist(i, j)=mean(temphist);
    end
end

chist = zeros(bin_height,channels);
for i = 1:bin_height
    for j = 1:channels
        temphist = rgbImg(bin_top+i-1, binImg(bin_top+i-1,:)>thres, j);
        chist(i, j)=mean(temphist);
    end
end

feature = [];
for j = 1:channels
    temphist = rhist(:,j);
    feature = [feature, mean(temphist), std(temphist)]; 
    temphist = chist(:,j);
    feature = [feature, mean(temphist), std(temphist)]; 
end

end