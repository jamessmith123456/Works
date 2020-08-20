function zeroMeanUnitVarImage = append_createZeroMeanUnitVarianceImage(rgbImg)

    [R C tmp] = size(rgbImg);
    image=reshape(rgbImg,R*C,3);

    meanRed = mean(image(:,1));
    meanGreen = mean(image(:,2));
    meanBlue = mean(image(:,3));

    stdRed = std(double(image(:,1)));
    stdGreen = std(double(image(:,2)));
    stdBlue = std(double(image(:,3)));

    zeroMeanUnitVarRed = (double(image(:,1))-meanRed)/double(stdRed);
    zeroMeanUnitVarGreen = (double(image(:,2))-meanGreen)/double(stdGreen);
    zeroMeanUnitVarBlue = (double(image(:,3))-meanBlue)/double(stdBlue);
    
    zeroMeanUnitVarImage(:,1) = zeroMeanUnitVarRed;
    zeroMeanUnitVarImage(:,2) = zeroMeanUnitVarGreen;
    zeroMeanUnitVarImage(:,3) = zeroMeanUnitVarBlue;
    

    zeroMeanUnitVarImage = reshape(zeroMeanUnitVarImage,R,C,3);

end