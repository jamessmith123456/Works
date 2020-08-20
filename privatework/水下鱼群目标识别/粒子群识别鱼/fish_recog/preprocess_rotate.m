function [ imgset_rot ] = preprocess_rotate( imgset, rotate_degree, flip )
%PREPROCESS_ROTATE Summary of this function goes here
%   Input: 
%       imgset: input image cells
%       rotate_degree: rotate degree array
%       flip: determine whether need flip, 1:yes, 2:no
%   Output:
%       imgset_rot: output image cells


img_num = length(imgset);

imgset_rot = cell(1, img_num);
for i = 1:img_num
    rawimg = imgset{i};
    rawimg_rot = imrotate(rawimg, -rotate_degree(i), 'bilinear');
    if flip(i) == 1
        for j = 1:size(rawimg_rot, 3)
        rawimg_rot(:,:,j) = fliplr(rawimg_rot(:,:,j));
        end
    end
    imgset_rot{i} = rawimg_rot;
end

end

