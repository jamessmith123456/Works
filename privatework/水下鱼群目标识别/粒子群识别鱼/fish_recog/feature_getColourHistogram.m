function [histR histG histB] = feature_getColourHistogram(rgbImage,maskImage)

    normalizedRgb = append_createZeroMeanUnitVarianceImage(rgbImage);
    [R C colourDepth] = size(normalizedRgb);
    
    if(colourDepth~=3)
        disp('Error: not an RGB image');
        return;
    end
    
    hist = [0,0,0];
    index = 0;
    for x=1:C
        for y=1:R
            if(maskImage(y,x)==255)
                index = index + 1;
                hist(index,1)=normalizedRgb(y,x,1);
                hist(index,2)=normalizedRgb(y,x,2);
                hist(index,3)=normalizedRgb(y,x,3);
            end
        end
    end
    
    edges = -2:0.05:2;
    
%    [R C]= size(hist);
%     normalisedRGB = zeros(R,2);
%     for i=1:R
%         normalisedRGB(i,1) = hist(i,1)/(hist(i,1)+hist(i,2)+hist(i,3));
%         normalisedRGB(i,2) = hist(i,2)/(hist(i,1)+hist(i,2)+hist(i,3));
%     end
%     norR=normalisedRGB(:,1);
%     norG=normalisedRGB(:,2);
    
    histR = histc(hist(:,1),edges);
    histG = histc(hist(:,2),edges);
    histB = histc(hist(:,3),edges);
   
    histR = histR/trapz(histR);
    histG = histG/trapz(histG);
    histB = histB/trapz(histB);
end