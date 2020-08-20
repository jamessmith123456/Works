function [ liklihood_valid, liklihood_invalid, model] = classify_GMM( feature_train, feature_test_valid, feature_test_invalid )

[samples, dimens] = size(feature_train);
max_comp = min(round(sqrt(dimens)), samples);
componentNum = gmm_mixtures4(feature_train',1,max_comp,0,1e-4,2);
[ model ] = classify_GMM_train( feature_train, 1:dimens, 1:dimens, componentNum );

liklihood_valid = -inf;
liklihood_invalid = -inf;
MAX_VALUE = 1000;
if ~isempty(model.obj)
    [ liklihood_valid ] = classify_GMM_predict( feature_test_valid, model );
    [ liklihood_invalid ] = classify_GMM_predict( feature_test_invalid, model );
    
    liklihood_valid = log(liklihood_valid+eps)-log(eps);
    liklihood_valid(isinf(liklihood_valid)) = MAX_VALUE;
    liklihood_invalid = log(liklihood_invalid+eps)-log(eps);
    liklihood_invalid(isinf(liklihood_invalid)) = MAX_VALUE;
end

end

