% This function takes two fish and a features index and returns the distance
% for that feature

% Fish1: Fish to be classified
% Fish2: Fish comparing against
% featureIndex: index of feature to use


function distance  = append_getFeatureDistances(feature1, feature2, featureIndex)
        distance = abs(feature1(featureIndex)-feature2(featureIndex));
end