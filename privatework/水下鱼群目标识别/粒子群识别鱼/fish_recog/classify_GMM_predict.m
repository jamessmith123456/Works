function [ liklihood ] = classify_GMM_predict( feature_predict, model )
%CLASSIFY_GMM_TRAIN Summary of this function goes here

liklihood = pdf(model.obj, feature_predict(:,model.indic));
end

