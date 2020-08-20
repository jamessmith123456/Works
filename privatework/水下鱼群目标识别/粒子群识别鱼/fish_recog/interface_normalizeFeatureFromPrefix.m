function [featuresRTN]= interface_normalizeFeatureFromPrefix(features, feature_mean, feature_std, feature_lower, feature_upper)

featuresRTN = [];
if nargin < 3
    featuresRTN = [];
    return;
end

if isempty(features)
    return;
end

[samples, dimens] = size(features);
[mean_samples, mean_dimens] = size(feature_mean);
[std_samples, std_dimens] = size(feature_std);
if 1 ~= mean_samples || 1 ~= std_samples || dimens ~= mean_dimens || dimens ~= std_dimens
    featuresRTN = [];
    return;
end

featuresRTN = (features - repmat(feature_mean, samples, 1) ) ./ (repmat(feature_std, samples, 1) + eps);
if nargin > 3
    featuresRTN = max( featuresRTN, repmat(feature_lower, samples, 1));
    featuresRTN = min( featuresRTN, repmat(feature_upper, samples, 1));
end

end