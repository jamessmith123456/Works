function binImg = append_cleanBinaryImage(binImg)

    if islogical(binImg)
        thres = 1;
        maxv = 1;
    else
        thres = 128;
        maxv = 255;
    end

    bin_neg = binImg < thres;
    binImg(bin_neg) = 0;
    binImg(~bin_neg) = maxv;
    
    labeled = bwlabel(binImg,4);
    stats = regionprops(labeled,'basic');
    [maxarea, maxindex] = max([stats.Area]);

    binImg(labeled ~= maxindex) = 0;
%     [H,W] = size(binImg);
%     for i=1:H
%         for j=1:W
%             if(labeled(i,j) ~= maxindex)
%                 binImg(i,j) = 0;
%             end
%         end
%     end
end