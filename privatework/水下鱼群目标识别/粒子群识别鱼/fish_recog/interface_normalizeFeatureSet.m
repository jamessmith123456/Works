function [featuresRTN, prefixValue]= interface_normalizeFeatureSet(features, prefixValue)
if nargin < 1
    featuresRTN = [];
    return;
end

[samples, dimens] = size(features);
lower_index = floor(samples * 0.025);
upper_index = ceil(samples * 0.975);

lower_value = zeros(1, dimens);
upper_value = zeros(1, dimens);
mean_value = zeros(1, dimens);
std_value = zeros(1, dimens);

if exist('prefixValue', 'var')
    lower_value = prefixValue.lower_value;
    upper_value = prefixValue.upper_value;
    mean_value = prefixValue.mean_value;
    std_value = prefixValue.std_value;
else
    for i = 1:dimens
        tmp_feature = features(:,i);
        tmp_feature_sort = sort(tmp_feature);
        
        lower_value(i) = tmp_feature_sort(lower_index);
        upper_value(i) = tmp_feature_sort(upper_index);
        
        tmp_feature(tmp_feature > upper_value(i)) = upper_value(i);
        tmp_feature(tmp_feature < lower_value(i)) = lower_value(i);
        
        mean_value(i) = mean(tmp_feature);
        std_value(i) = std(tmp_feature) + eps;
        
        lower_value(i) = (lower_value(i) - mean_value(i)) / std_value(i);
        upper_value(i) = (upper_value(i) - mean_value(i)) / std_value(i);
        
%         limitation = 20;
%         lower_value(i) = -limitation * std_value(i);
%         upper_value(i) = limitation * std_value(i);
    end
end

[featuresRTN]= interface_normalizeFeatureFromPrefix(features, mean_value, std_value, lower_value, upper_value);
%[featuresRTN]= interface_normalizeFeatureFromPrefix(features, mean_value, std_value);

prefixValue.lower_value = lower_value;
prefixValue.upper_value = upper_value;
prefixValue.mean_value = mean_value;
prefixValue.std_value = std_value;

end