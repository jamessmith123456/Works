function [theta, details] = append_orienCurve(binImg)

    theta = [];
    details = [];
    
    binImg = append_cleanBinaryImage(binImg);

    contK = feature_curvecorner(binImg);
    
    if isempty(contK)
        return;
    end
    
    contK(:,3) = abs(contK(:,3) - mean(contK(:,3)));
    %contK(2:end,3) = abs(contK(1:end-1,3) - contK(2:end,3)); contK(1,3) = 0;
    
    weightcenterx = sum(contK(:,2).*contK(:,3))/sum(contK(:,3));
    weightcentery = sum(contK(:,1).*contK(:,3))/sum(contK(:,3));
    
    warning off;
    centr = regionprops(binImg,'centroid');
    warning on;
    centx = centr(end).Centroid(1);
    centy = centr(end).Centroid(2);
    
    offsetx = centx - weightcenterx;
    offsety = centy - weightcentery;
    
    [theta, drop] = cart2pol(offsetx, offsety);
    theta = theta * 180 / pi;
    theta(isnan(theta))=0;
    
    details.cx = centx;
    details.cy = centy;
    details.wx = weightcenterx;   
    details.wy = weightcentery;
    
%     figure(2);
%     imshow(binImg);
%     hold on;
%     plot(centx, centy, 'r*');
%     plot(weightcenterx, weightcentery, 'r+');
%     hold off;

end