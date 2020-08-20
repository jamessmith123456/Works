function [ norcolor ] = append_normalizecolor( rgbimg_ui, maskimg, showoption)
%APPEND_NORMALIZECOLOR Summary of this function goes here
%   Detailed explanation goes here

rgbimg = double(rgbimg_ui);
[m,n,c] = size(rgbimg);
if c ~= 3
    fprintf('non color image');
    return;
end

norcolor = [];
count = 0;
for i = 1:m
    for j = 1:n
        r = rgbimg(i,j,1);
        g = rgbimg(i,j,2);
        b = rgbimg(i,j,3);
        
        rgbsum = r+g+b;
        if 0 < rgbsum && 128 < maskimg(i,j)
            count = count + 1;
            norcolor(count,1) = r / rgbsum;
            norcolor(count,2) = g / rgbsum;
        end
    end
end

if showoption
    
figure(1);

subplot(1,3,1);
imshow(rgbimg_ui);

subplot(1,3,2);
imshow(maskimg);

subplot(1,3,3);
plot(norcolor(:,1), norcolor(:,2), '+');

end

end

