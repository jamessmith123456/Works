function rgbImage = append_getSpecificImage(rgbImage,maskImage)

rgbImage1 = rgbImage;
    [R C colourDepth] = size(rgbImage);
    if(colourDepth~=3)
        disp('Error: not an RGB image');
        return;
    end
    
    for x=1:C
        for y=1:R
            if(maskImage(y,x)<128)
                rgbImage(y,x,1) = 0;
                rgbImage(y,x,2) = 0;
                rgbImage(y,x,3) = 0;
            else
                if((rgbImage(y,x,1)==0)&&(rgbImage(y,x,2)==0)&&(rgbImage(y,x,3)==0))
                    rgbImage(y,x,1)=1;
                    rgbImage(y,x,2)=1;
                    rgbImage(y,x,3)=1;
                end
            end
        end
    end
%     imshow(rgbImage);
end