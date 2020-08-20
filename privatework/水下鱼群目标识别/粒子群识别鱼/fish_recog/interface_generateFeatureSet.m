function [features, feature_index]= interface_generateFeatureSet(rgbImg, binImg, use_rotate, self_normalize, save_result_file)

features = [];
feature_index = [];

if nargin < 4
    self_normalize = 0;
end

if nargin < 3
    use_rotate = 1;
end

if nargin < 2
    return;
end

if isempty(rgbImg) || isempty(binImg)
    return;
end

%extract feature
sample_num = length(rgbImg);
reverseStr = '';
fprintf('%s start to process %d samples\n', append_timeString(), sample_num);
for i = 1:sample_num
    string_printf = sprintf('%s feature extraction: Processing %d/%d\n', append_timeString(), i, sample_num);
    fprintf([reverseStr, string_printf]);
    reverseStr = repmat(sprintf('\b'), 1, length(string_printf));
    
    [features(i,:), feature_index] = feature_generateFeatureVector(rgbImg{i},binImg{i}, use_rotate);
    feature_index = feature_index(1,:);
    
    if exist('save_result_file', 'var') && 0 == mod(i, 1000)
        save(save_result_file, 'features', 'feature_index');
    end
end

if self_normalize == 1
    features= interface_normalizeFeatureSet(features);
end
end