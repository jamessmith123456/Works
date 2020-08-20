function [rgbRot, binRot, theta_rot]= append_orienFish(rgbImg, binImg)

channel = size(binImg, 3);
if 1< channel
    fprintf('only accept grey image');
    return;
end

if islogical(binImg)
    binImg = uint8(binImg);
    binImg = binImg * 255;
end


binImg = append_cleanBinaryImage(binImg);

warning off;
orienset = regionprops(binImg,'orientation');
theta_rot_ellipse = -orienset(end).Orientation;
warning on;

[theta_curve]=append_orienCurve(binImg);
if isempty(theta_curve)
    theta_curve = 0;
end

theta_rot_ellipse = mod(theta_rot_ellipse - theta_curve + 90, 180) - 90 + theta_curve;

if abs(theta_rot_ellipse - theta_curve) < 30
    theta_rot = theta_rot_ellipse;
else
    theta_rot = theta_curve;
end

rgbRot = append_rotateFish_d(rgbImg, theta_rot);

binRot = append_rotateFish_d(binImg, theta_rot);

end

