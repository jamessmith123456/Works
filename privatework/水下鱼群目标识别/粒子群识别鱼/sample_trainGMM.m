addpath('./fish_recog');

%load testing image data
tmp_data = load('features.mat');

%example of training GMM for class 1.
feature_train = tmp_data.features(tmp_data.class_id==1,:);
[samples, dimens]=size(feature_train);
componentNum = gmm_mixtures4(feature_train',1,7,0,1e-4,2);
[ model ] = classify_GMM_train( feature_train, 1:dimens, 1:dimens, componentNum );