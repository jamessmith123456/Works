function [hier_arch] = classify_HierSVM_train(feature_train, classid_train, featurei, hier_arch, classifier, option_add)

if ~exist('option_add', 'var')
    option_add = '';
end

b_use_node_rejection = 1;
% this will lead a bug: sub branch does not have Node_Rejection file.
% if isfield(hier_arch, 'Node_Rejection')
%     b_use_node_rejection = hier_arch.Node_Rejection;
% end

hier_arch.attribute_nodeRejection = b_use_node_rejection;
hier_arch.feature_id = featurei;

number_node = length(hier_arch.branch_class);

hier_arch.Input_classid_train = classid_train;
hier_arch.Input_classid_set = [];
for i = 1:number_node
    tmp_set = hier_arch.branch_class{i};
    hier_arch.Input_classid_set = [hier_arch.Input_classid_set, reshape(tmp_set, 1, length(tmp_set))];
end

number_train = length(classid_train);
Node_classid_train = -1 * ones(number_train, 1);
for i = 1:number_node
    Node_classid_train(ismember(classid_train, hier_arch.branch_class{i})) = i;
end

Node_featureSet = unique(featurei);
if isfield(hier_arch, 'Subfeature')
    Node_featureSet = hier_arch.Subfeature;
end
featureSet_train = feature_train(:, ismember(featurei, Node_featureSet));

if ~b_use_node_rejection
    minus_index = Node_classid_train == -1;
    featureSet_train = featureSet_train(~minus_index, :);
    Node_classid_train = Node_classid_train(~minus_index);
end

[hier_arch.model] = classify_SVM_train(featureSet_train, Node_classid_train, classifier, option_add);

for i = 1:number_node
    if ~isempty(hier_arch.branch_link{i})
        [hier_arch.branch_link{i}] = classify_HierSVM_train(feature_train, classid_train, featurei, hier_arch.branch_link{i}, classifier);
    end
end

if isfield(hier_arch, 'Root')
    %check feature subset
    rootfeatureSet = unique(featurei);
    if isfield(hier_arch.Root, 'Subfeature')
        rootfeatureSet = hier_arch.Root.Subfeature;
    end
    RootfeatureSet_train = feature_train(:, ismember(featurei, rootfeatureSet));
    [hier_arch.Root.model] = classify_SVM_train(RootfeatureSet_train, classid_train, classifier, option_add);
end


%GMM rejection
if isfield(hier_arch, 'SpeciesSubfeature')
    classSet = unique(classid_train);
    cellSize = length(hier_arch.SpeciesSubfeature);
    
    for i = 1:length(classSet)
        candidateSpecies = classSet(i);
        if candidateSpecies > cellSize || isempty(hier_arch.SpeciesSubfeature{candidateSpecies})
            featureSub = unique(featurei);
        else
            featureSub = hier_arch.SpeciesSubfeature{candidateSpecies};
        end
        
        GMMcomponent = 1:5;
        tmp_indic = classid_train == classSet(i);
        for j = GMMcomponent
            hier_arch.GMMmodel{j, candidateSpecies} = classify_GMM_train( feature_train(tmp_indic, :), featurei, featureSub, j);
        end
    end
end
end