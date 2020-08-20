function [headImg tailImg topImg bottomImg half_headImg half_tailImg] = append_seperateFish(binImg_rot)
%APPEND_SEPERATEFISH Summary of this function goes here
%   Detailed explanation goes here
warning off;
centr = regionprops(binImg_rot,'centroid');
centx = int32(centr(255).Centroid(1));
centy = int32(centr(255).Centroid(2));

box_array = regionprops(binImg_rot,'boundingbox');
box_mask = box_array(255).BoundingBox;

half_left = round(box_mask(1)+box_mask(3)/4);
half_right = round(box_mask(1)+box_mask(3)*3/4);

headImg = binImg_rot; headImg(:,1:centx) = 0;
tailImg = binImg_rot; tailImg(:,centx:end) = 0;
topImg = binImg_rot; topImg(centy:end,:) = 0;
bottomImg = binImg_rot; bottomImg(1:centy,:) = 0;
half_headImg = binImg_rot; half_headImg(:,1:half_right) = 0;
half_tailImg = binImg_rot; half_tailImg(:,half_left:end) = 0;

warning on;
end

