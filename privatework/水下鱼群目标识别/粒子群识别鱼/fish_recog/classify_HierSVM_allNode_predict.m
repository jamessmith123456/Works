function [scoreSet] = classify_HierSVM_allNode_predict(hier_arch, feature_test, scoreSet)

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

scoreSet{end+1,1}=hier_arch.node_id;
[hier_arch.Node_predict, scoreSet{end,2}] = classify_SVM_predict(hier_arch.model, featureSet_test);

for i = 1:number_node
    if ~isempty(hier_arch.branch_link{i})% && sum(tmp_indicates)
        [scoreSet] = classify_HierSVM_allNode_predict(hier_arch.branch_link{i}, feature_test, scoreSet);
    end
end


end