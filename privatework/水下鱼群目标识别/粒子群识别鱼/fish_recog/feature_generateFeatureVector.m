function [features, featurei] = feature_generateFeatureVector(rgbimg, binimg, use_rotate)
[rgbimg, binimg] = append_shrinkImage(rgbimg, binimg);

if use_rotate
    [rgbImg_rot, binImg_rot, theta_rot_drop]= append_orienFish(rgbimg, binimg);
else
    rgbImg_rot = rgbimg;
    binImg_rot = binimg;
end

binImg_rot = append_cleanBinaryImage(binImg_rot);

[headImg tailImg topImg bottomImg half_headImg half_tailImg] = append_seperateFish(binImg_rot);
features = [];
featurei = [];

[headNorR, headNorG] = feature_getNormalizedRG(rgbImg_rot, headImg); 
[tailNorR, tailNorG] = feature_getNormalizedRG(rgbImg_rot, tailImg);
[topNorR, topNorG] = feature_getNormalizedRG(rgbImg_rot, topImg);
[bottomNorR, bottomNorG] = feature_getNormalizedRG(rgbImg_rot, bottomImg);
[FishNorR, FishNorG] = feature_getNormalizedRG(rgbImg_rot, binImg_rot);

headh = append_hcolor(rgbImg_rot, headImg); 
tailh = append_hcolor(rgbImg_rot,tailImg);
toph = append_hcolor(rgbImg_rot,topImg);
bottomh = append_hcolor(rgbImg_rot,bottomImg);
Fishh = append_hcolor(rgbImg_rot,binImg_rot);

offset = 0.2;
headh = mod(headh+offset, 1); 
tailh = mod(tailh+offset, 1); 
toph = mod(toph+offset, 1); 
bottomh = mod(bottomh+offset, 1); 
Fishh = mod(Fishh+offset, 1); 

feature_index = 1;

%normalized color red
histrange = 0:0.02:1;

temphist = histc(headNorR, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailNorR, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(topNorR, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomNorR, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(FishNorR, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

%normalized color green
histrange = 0:0.02:1;

temphist = histc(headNorG, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailNorG, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(topNorG, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomNorG, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(FishNorG, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

% h component in HSV model
temphist = histc(headh, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailh, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(toph, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomh, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(Fishh, histrange);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

%normalized color red _ new
histrange_nr = [0, 0.01742,0.125,0.1884,0.226,0.26,0.2934,0.328,0.372,0.4512, 1];

temphist = histc(headNorR, histrange_nr);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailNorR, histrange_nr);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(topNorR, histrange_nr);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomNorR, histrange_nr);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(FishNorR, histrange_nr);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

%normalized color green _ new
histrange_ng = [0, 0.298,0.3276,0.35,0.3734,0.404,0.4334,0.466,0.505,0.59, 1];

temphist = histc(headNorG, histrange_ng);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailNorG, histrange_ng);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(topNorG, histrange_ng);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomNorG, histrange_ng);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(FishNorG, histrange_ng);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

% h component in HSV model _ new
histrange_h = [0, 0.2246,0.291,0.463,0.5648,0.6034,0.6248,0.6422,0.675,0.7164, 1];
%histrange_h = [0, 0.0602,0.1736,0.3322,0.3898,0.4162,0.434,0.4548,0.5002,0.5588, 1];

temphist = histc(headh, histrange_h);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(tailh, histrange_h);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(toph, histrange_h);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(bottomh, histrange_h);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

temphist = histc(Fishh, histrange_h);
temphist = temphist / sum(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

%curvature shape
[cstd, curve_tail_ratio] = getCstdRatio(binImg_rot);
[features, featurei, feature_index] = mergefeature(cstd, features, featurei, feature_index, 1);
[features, featurei, feature_index] = mergefeature(curve_tail_ratio, features, featurei, feature_index, 1);

%fish desity statistic
temphist = feature_densityfeature(rgbImg_rot, binImg_rot);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

%Co-occurance matrix
CoOccurance = feature_getCoOccurrenceMatrix(rgbImg_rot,binImg_rot);
[R,C] = size(CoOccurance);
for i = 1:C
    temphist =  CoOccurance(:,i);
    [features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);
end

%moment invirant
[feature.Headci1 feature.Headci2 feature.Headci3 feature.Headci4 feature.Headci5 feature.Headci6] = feature_getComplexMoments(headImg);
temphist = [feature.Headci1; feature.Headci2; feature.Headci3; feature.Headci4; feature.Headci5; feature.Headci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Tailci1 feature.Tailci2 feature.Tailci3 feature.Tailci4 feature.Tailci5 feature.Tailci6] = feature_getComplexMoments(tailImg);
temphist = [feature.Tailci1; feature.Tailci2; feature.Tailci3; feature.Tailci4; feature.Tailci5; feature.Tailci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Topci1 feature.Topci2 feature.Topci3 feature.Topci4 feature.Topci5 feature.Topci6] = feature_getComplexMoments(topImg);
temphist = [feature.Topci1; feature.Topci2; feature.Topci3; feature.Topci4; feature.Topci5; feature.Topci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Bottomci1 feature.Bottomci2 feature.Bottomci3 feature.Bottomci4 feature.Bottomci5 feature.Bottomci6] = feature_getComplexMoments(bottomImg);
temphist = [feature.Bottomci1; feature.Bottomci2; feature.Bottomci3; feature.Bottomci4; feature.Bottomci5; feature.Bottomci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Half_headci1 feature.Half_headci2 feature.Half_headci3 feature.Half_headci4 feature.Half_headci5 feature.Half_headci6] = feature_getComplexMoments(half_headImg);
temphist = [feature.Half_headci1; feature.Half_headci2; feature.Half_headci3; feature.Half_headci4; feature.Half_headci5; feature.Half_headci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Half_tailci1 feature.Half_tailci2 feature.Half_tailci3 feature.Half_tailci4 feature.Half_tailci5 feature.Half_tailci6] = feature_getComplexMoments(half_tailImg);
temphist = [feature.Half_tailci1; feature.Half_tailci2; feature.Half_tailci3; feature.Half_tailci4; feature.Half_tailci5; feature.Half_tailci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);

[feature.Wholeci1 feature.Wholeci2 feature.Wholeci3 feature.Wholeci4 feature.Wholeci5 feature.Wholeci6] = feature_getComplexMoments(binImg_rot);
temphist = [feature.Wholeci1; feature.Wholeci2; feature.Wholeci3; feature.Wholeci4; feature.Wholeci5; feature.Wholeci6];
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 1);


%pyramid of histogram of orientation
phog_bin = 8;
phog_angle = 360;
phog_L=3;
phog_feature = feature_anna_phog(binImg_rot,phog_bin,phog_angle,phog_L);
for i = 1:length(phog_feature)
    temphist = phog_feature{i};
    [features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);
end

%fourier descriptor
fourier_size = 15;
temphist = feature_getFourierDescriptors(binImg_rot, fourier_size);
%temphist = (temphist-mean(temphist))/std(temphist);
[features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, 0);

%gabor filter
gabor_head = feature_gaborfeature(rgbImg_rot, headImg); 
[features, featurei, feature_index] = mergefeature(gabor_head, features, featurei, feature_index, 1);

gabor_tail = feature_gaborfeature(rgbImg_rot, tailImg);
[features, featurei, feature_index] = mergefeature(gabor_tail, features, featurei, feature_index, 1);

gabor_top = feature_gaborfeature(rgbImg_rot, topImg);
[features, featurei, feature_index] = mergefeature(gabor_top, features, featurei, feature_index, 1);

gabor_bottom = feature_gaborfeature(rgbImg_rot, bottomImg);
[features, featurei, feature_index] = mergefeature(gabor_bottom, features, featurei, feature_index, 1);

gabor_Fish = feature_gaborfeature(rgbImg_rot, binImg_rot);
[features, featurei, feature_index] = mergefeature(gabor_Fish, features, featurei, feature_index, 1);      

%AMI features
ami_Fish = feature_AffineMomentInvariant( binImg_rot );
[features, featurei, feature_index] = mergefeature(ami_Fish, features, featurei, feature_index, 1);      

ami_head = feature_AffineMomentInvariant( headImg );
[features, featurei, feature_index] = mergefeature(ami_head, features, featurei, feature_index, 1);      

ami_tail = feature_AffineMomentInvariant( tailImg );
[features, featurei, feature_index] = mergefeature(ami_tail, features, featurei, feature_index, 1);      

ami_top = feature_AffineMomentInvariant( topImg );
[features, featurei, feature_index] = mergefeature(ami_top, features, featurei, feature_index, 1);      

ami_bottom = feature_AffineMomentInvariant( bottomImg );
[features, featurei, feature_index] = mergefeature(ami_bottom, features, featurei, feature_index, 1);      

ami_half_head = feature_AffineMomentInvariant( half_headImg );
[features, featurei, feature_index] = mergefeature(ami_half_head, features, featurei, feature_index, 1);      

ami_half_tail = feature_AffineMomentInvariant( half_tailImg );
[features, featurei, feature_index] = mergefeature(ami_half_tail, features, featurei, feature_index, 1);      

%find half_tail / half_head area ratio
MAR_half_head = feature_MaskAreaRatio( half_headImg,  binImg_rot);
[features, featurei, feature_index] = mergefeature(MAR_half_head, features, featurei, feature_index, 1);      

MAR_half_tail = feature_MaskAreaRatio( half_tailImg,  binImg_rot);
[features, featurei, feature_index] = mergefeature(MAR_half_tail, features, featurei, feature_index, 1);      

%end of feature extr20action
features(isnan(features))=0;
end

function [cstd, curve_tail_ratio] = getCstdRatio(binImg)

cstd = 0;
curve_tail_ratio = 0;
K_cout = feature_curvecorner(binImg);
if isempty(K_cout)
    return;
end

extremum_all = find(K_cout(:,4)==1);
dista = zeros(length(extremum_all),1);
for ii = 2:length(extremum_all)-1
    prev = extremum_all(ii-1);
    cur = extremum_all(ii);
    next = extremum_all(ii+1);
    dista(ii) = abs(K_cout(prev, 3) - K_cout(cur, 3)) + abs(K_cout(cur, 3) - K_cout(next, 3));
end
[drop, maxi] = max(dista);
prev = extremum_all(max(1,maxi-1));
next = extremum_all(min(maxi+1, length(extremum_all)));
distb = std(K_cout(prev : next, 3));
cstd = distb/std(K_cout(:,3));

cent_x = round((K_cout(prev, 2)+K_cout(next, 2))/2);
half_tailImg = binImg; half_tailImg(:,cent_x:end) = 0;
curve_tail_ratio = feature_MaskAreaRatio( half_tailImg,  binImg);

end

function [features, featurei, feature_index] = mergefeature(temphist, features, featurei, feature_index, ishist)
% if length(temphist) > 1
%     temphist = temphist - min(temphist);
%     if eps < sum(temphist)
%         temphist = temphist / sum(temphist);
%     end
% end

%normalize =0;

add_length = length(temphist);

features(end+1:end+add_length) = temphist;
featurei(:, end+1:end+add_length) = repmat([feature_index;ishist], 1, add_length);
feature_index = feature_index + 1;
end

function [features, featurei, feature_index] = mergefeature_individual(temphist, features, featurei, feature_index, ishist)
for i = 1:length(temphist)
    [features, featurei, feature_index] = mergefeature(temphist(i), features, featurei, feature_index, ishist);      
end
end