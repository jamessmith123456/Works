function [norR norG] = feature_getNormalizedRG(rgbImage,maskImage)

    normalizedRgb = rgbImage;
    [R C colourDepth] = size(normalizedRgb);
    
    if(colourDepth~=3)
        disp('Error: not an RGB image');
        return;
    end
    
    if islogical(maskImage)
        thres = 1;
    else 
        thres = 128;
    end
    
    hist = [0,0,0];
    index = 0;
    for x=1:C
        for y=1:R
            if(maskImage(y,x)>=thres)
                index = index + 1;
                hist(index,1)=normalizedRgb(y,x,1);
                hist(index,2)=normalizedRgb(y,x,2);
                hist(index,3)=normalizedRgb(y,x,3);
            end
        end
    end
    
   [R C]= size(hist);
    normalisedRGB = zeros(R,2);
    for i=1:R
        weightall = max(1, hist(i,1)+hist(i,2)+hist(i,3));
        normalisedRGB(i,1) = hist(i,1)/weightall;
        normalisedRGB(i,2) = hist(i,2)/weightall;
    end
    norR=normalisedRGB(:,1);
    norG=normalisedRGB(:,2);
end