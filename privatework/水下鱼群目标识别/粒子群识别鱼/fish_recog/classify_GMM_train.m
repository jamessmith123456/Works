function [ model ] = classify_GMM_train( feature_train, featurei, subfeature_set, component_num )
%CLASSIFY_GMM_TRAIN Summary of this function goes here
indic = ismember(featurei, subfeature_set);
feature_train = feature_train(:,indic);

tmp_indic = var(feature_train)>1e-3 & var(feature_train)<1e+6;
feature_train = feature_train(:,tmp_indic);
tmp_index = find(indic);
indic(tmp_index(~tmp_indic)) = 0;

[samples, dimens] = size(feature_train);
model.obj = [];
model.indic = [];
if samples <= dimens || dimens < 1 || samples <= component_num
    return;
%    feature_train = repmat(feature_train, ceil(dimens/samples), 1);
end

% tmp_indic = std(feature_train)>0.01;
% feature_train = feature_train(:,tmp_indic);
% tmp_index = find(indic);
% indic(tmp_index(~tmp_indic)) = 0;

model.obj = gmdistribution.fit(feature_train, component_num, 'Regularize',1e-4,'Options',statset('MaxIter',1500), 'CovType','full');
model.indic = indic;

% NComponents = model.NComponents;
% prior=cell(NComponents,1);
% mean=cell(NComponents,1);
% Sigma=cell(NComponents,1);
% for n = 1:NComponents
% prior{n,1}=model.PComponents(n); % prior
% mean{n,1}=model.mu(n,:); % mean
% Sigma{n,1}=model.Sigma(:,:,n); % Sigma
% end


end

