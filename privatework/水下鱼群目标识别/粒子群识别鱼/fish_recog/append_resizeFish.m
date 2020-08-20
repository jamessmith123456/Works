function [ rgbImg, binImg ] = append_resizeFish( rgbImg, binImg )
%APPEND_RESIZEFISH Summary of this function goes here
%   Detailed explanation goes here
sample_num = length(rgbImg);

resize_to = 100;

for i = 1:sample_num
    
    ctime = clock();
    fprintf('%d:%d:%d:%d:%d:%f resizing fish: Processing %d/%d\n', ctime(1), ctime(2), ctime(3), ctime(4), ctime(5), ctime(6), i, sample_num);

    tmp_binImg = append_cleanBinaryImage(binImg{i});
    if ~islogical(tmp_binImg)
        tmp_binImg(tmp_binImg > 128) = 255;
    end
    tmp_box = regionprops(tmp_binImg, 'BoundingBox');
    tmp_box = max(1, floor(tmp_box(end).BoundingBox));
    
    rgbImg{i} = imresize(rgbImg{i}(tmp_box(2):tmp_box(2)+tmp_box(4), tmp_box(1):tmp_box(1)+tmp_box(3), :), [resize_to, resize_to], 'bilinear');
    binImg{i} = imresize(binImg{i}(tmp_box(2):tmp_box(2)+tmp_box(4), tmp_box(1):tmp_box(1)+tmp_box(3), :), [resize_to, resize_to], 'bilinear');
end

end

