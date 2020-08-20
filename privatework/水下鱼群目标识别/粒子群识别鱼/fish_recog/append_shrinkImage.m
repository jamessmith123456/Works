function [rgbimg, binimg] = append_shrinkImage(rgbimg, binimg)

shrink_thres = 256;
while size(binimg, 1) > shrink_thres && size(binimg, 2) > shrink_thres
    rgbimg = imresize(rgbimg, 0.5);
    binimg = imresize(binimg, 0.5);
end

end