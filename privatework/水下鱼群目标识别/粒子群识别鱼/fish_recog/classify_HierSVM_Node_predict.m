function [hier_arch] = classify_HierSVM_Node_predict(hier_arch, feature_test)

hier_arch.Node_predict = [];
hier_arch.Node_scores = [];

number_node = length(hier_arch.branch_class);
number_test = size(feature_test,1);
if 1 > number_test
    return;
end

featurei = hier_arch.feature_id;
Node_featureSet = unique(featurei);
if isfield(hier_arch, 'Subfeature')
    Node_featureSet = hier_arch.Subfeature;
end
featureSet_test = feature_test(:, ismember(featurei, Node_featureSet));

[hier_arch.Node_predict, hier_arch.Node_scores] = classify_SVM_predict(hier_arch.model, featureSet_test);

for i = 1:number_node
    tmp_indicates = (hier_arch.Node_predict==i);
    if ~isempty(hier_arch.branch_link{i})% && sum(tmp_indicates)
        tmp_feature = feature_test(tmp_indicates, :);
        [hier_arch.branch_link{i}] = classify_HierSVM_Node_predict(hier_arch.branch_link{i}, tmp_feature);
    end
end


end