function [ hcolor ] = append_hcolor( rgbimg, maskimg, showoption)
%APPEND_NORMALIZECOLOR Summary of this function goes here
%   Detailed explanation goes here

%rgbimg = double(rgbimg_ui);

if nargin < 3
    showoption = 0;
end

hsvimg = rgb2hsv(rgbimg);
[m,n,c] = size(rgbimg);
if c ~= 3
    fprintf('non color image');
    return;
end

if islogical(maskimg)
    thres = 1;
else
    thres = 128;
end

hcolor = [];
count = 0;
for i = 1:m
    for j = 1:n
        h = hsvimg(i,j,1);

        if thres <= maskimg(i,j)
            count = count + 1;
            hcolor(count,1) = h;
        end
    end
end

if showoption
    
figure(1);

subplot(1,3,1);
imshow(rgbimg);

subplot(1,3,2);
imshow(maskimg);

subplot(1,3,3);
plot(hcolor, '+');

end

end

