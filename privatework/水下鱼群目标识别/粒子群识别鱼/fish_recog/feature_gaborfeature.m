function [ features ] = feature_gaborfeature( srcImg, binImg )

warning off;

[m,n,channels] = size(srcImg);
if channels > 1
    srcImg = rgb2gray(srcImg);
end

if islogical(binImg)
    thres = 1;
else
    thres = 128;
end
%thres = ceil(max(max(binImg))/2);
%srcImg(binImg<thres)=0;

feature_mean = [];
feature_std = [];

scale = 2:2:8;
orient = 0:pi/4:3*pi/4;

for s = scale
    for o = orient
        gabout = append_gaborfilter(srcImg, s, o);
        
        gabout_valid = gabout(binImg>= thres);
        
        gabout_valid(gabout_valid<intmin('uint8'))=intmin('uint8');
        gabout_valid(gabout_valid>intmax('uint8'))=intmax('uint8');
        
        ghist = histc(uint8(gabout_valid), 0:255);
        
        mean_i = 0:255;
        ghist = reshape(ghist, length(ghist), 1);
        mean_v = mean_i * ghist / sum(ghist);
        feature_mean = [feature_mean; mean_v];
        
        std_i = ((0:255) - mean_v) .^ 2;
        std_v = std_i * ghist / sum(ghist);
        feature_std = [feature_std; std_v]; 
    end
end

%feature_mean = (feature_mean - mean(feature_mean)) / std(feature_mean);
%feature_std = (feature_std - mean(feature_std)) / std(feature_std);

features = [feature_mean; feature_std];

warning on;
end

