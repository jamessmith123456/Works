function [image_accpet] = append_acceptImage(fishImg, binImg)
image_accpet = [];
if length(fishImg) ~= length(binImg)
    return;
end

image_accpet = ones(length(binImg), 1);
for i = 1:length(binImg)
    binImg_rot = append_cleanBinaryImage(binImg{i});
    stats = regionprops(binImg_rot,'basic');
    [maxarea, maxindex] = max([stats.Area]);
    bounding_box = stats(maxindex).BoundingBox;
    
    if bounding_box(3) < 10 || bounding_box(4) < 10
        image_accpet(i) = 0;
    end
end
end