function [rt_predict, rt_scores, hier_arch] = classify_HierSVM_predict(feature_test, hier_arch)

%
% Xuan (Phoenix) Huang                Xuan.Huang@ed.ac.uk
% Edinburgh, United Kingdom                      Feb 2012 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%classify_HierSVM

rt_predict = [];
rt_scores = [];

if nargin < 2
	tmp_datadefault = load('data_default');
    hier_arch = tmp_datadefault.hier_model;
end

if isempty(feature_test)
    return;
end

featurei = hier_arch.feature_id;

%hierarch classification for each node
[hier_arch] = classify_HierSVM_Node_predict(hier_arch, feature_test);

%hierarch classification for all nodes
[hier_arch.allNodeScores] = classify_HierSVM_allNode_predict(hier_arch, feature_test, []);

rt_predict = hier_arch.Node_predict;
rt_scores = hier_arch.Node_scores;

%summarize node result
if isfield(hier_arch, 'b_isTree')
    [hier_arch] = classify_HierSVM_TreePredict(hier_arch);
    rt_predict = hier_arch.Tree_predict;
    rt_scores = hier_arch.Tree_Score_history;
end

if isfield(hier_arch, 'Root')
    %determine retry samples
    Retry_index = find(rt_predict == -1);
    
    if 0 < length(Retry_index)
        %check feature subset
        rootfeatureSet = unique(featurei);
        if isfield(hier_arch.Root, 'Subfeature')
            rootfeatureSet = hier_arch.Root.Subfeature;
        end
        RootfeatureSet_test = feature_test(Retry_index, ismember(featurei, rootfeatureSet));
    
        hier_arch.RootRetry_index = Retry_index;
        [hier_arch.RootRetry_predict, hier_arch.RootRetry_scores] = classify_SVM_predict(hier_arch.Root.model, RootfeatureSet_test);
        
        %convert score orders
        tmp_scores = cell2mat(hier_arch.RootRetry_scores);
        [tmp_indic, tmp_order]=ismember(sort(hier_arch.Root.model.models.Label), hier_arch.Root.model.models.Label);
        hier_arch.RootRetry_scores = mat2cell(tmp_scores(:, tmp_order), ones(size(tmp_scores, 1),1), size(tmp_scores, 2));
        %convert score orders
        
        rt_predict(Retry_index) = hier_arch.RootRetry_predict;
    end
end

if isfield(hier_arch, 'GMMmodel')  
[GMMcomponent, candidateSpecies] = size(hier_arch.GMMmodel);
for i = 1:candidateSpecies
    for j = 1:GMMcomponent
        if isempty(hier_arch.GMMmodel{j, i})
            continue;
        end
        hier_arch.GMMResult{j}(:,i) = classify_GMM_predict(feature_test, hier_arch.GMMmodel{j, i});
    end
end
%clear hier_arch.GMMmodel;
end

rt_scores = append_hierScoreToFlat(hier_arch, rt_scores);
if isfield(hier_arch, 'RootRetry_scores')
    rt_scores(hier_arch.RootRetry_index) =  hier_arch.RootRetry_scores;
end