function [ HierTree ] = append_hierFeatureSelection( HierTree, features, feature_id, classid, traj, filesavename_prefix)

classifier = 2;
standard = 3;
fold = 5;

maxfeature = length(unique(feature_id));

filesavename = sprintf('%s_Node%d', filesavename_prefix, HierTree.node_id);
tmp_classid = classid;
for i = 1:length(HierTree.branch_class)
    tmp_classid(ismember(classid, HierTree.branch_class{i})) = i;
end
tmp_classid(~ismember(classid, HierTree.node_classSet)) = -1;
[featureSubset, scoreResult] = classify_featureselection_fw(features, feature_id, tmp_classid, traj, maxfeature, standard, fold, classifier, filesavename);
[mv, mi] = max(scoreResult(:, standard));
HierTree.Subfeature = featureSubset(1:mi);

filesavename = sprintf('%s_Tree%d', filesavename_prefix, HierTree.node_id);
save(filesavename, 'HierTree');

for i = 1:length(HierTree.branch_link)
    if ~isempty(HierTree.branch_link{i})
        [HierTree.branch_link{i}] = append_hierFeatureSelection(HierTree.branch_link{i}, features, feature_id, classid, traj, filesavename_prefix);
    end
end

end