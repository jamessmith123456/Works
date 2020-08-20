function [ image_ori ] = append_rotateFish_d(image, rotates)

rotate = mod(rotates + 180, 360) - 180;

flip = 0;
if abs(rotate)>90
    flip = 1;
    rotateset = [180 + rotate, -180 + rotate];
    [minv mini] = min(abs(rotateset));
    rotate = rotateset(mini);    
end

image_ori = imrotate(image, rotate, 'bilinear');
channel = size(image_ori, 3);

if flip
    for i = 1:channel
        image_ori(:,:,i) = fliplr(image_ori(:,:,i));
    end
end

end

