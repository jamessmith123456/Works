function [ hier ] = append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, class_set, node_id, test_set, cv_one_fold)
%APPEND_CONSTRUCTRECURSIVENODE Summary of this function goes here

hier = [];
if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

if ~exist('test_set','var')
    test_set = 0;
end


class_size = length(class_set);
if class_size < 4
    return;
end

hier.node_id = node_id;
%hier.class_id = classid;

class_size_left = floor(class_size / 2);

class_subset_index = ismember(classid, class_set);
sample_subset = sum(class_subset_index);
feature_subset = feature(class_subset_index, :);
classid_subset = classid(class_subset_index);
traj = traj(class_subset_index);

max_score = -0.1;
max_pattern = 0;
upper_size = (2 ^ class_size)-1;
for i = 1:upper_size
    binary_pattern = de2bi(i, class_size) == 1;
    if class_size_left ~= sum(binary_pattern)
        continue;
    end
    
    tmp_classid = ones(sample_subset, 1);
    tmp_classid(ismember(classid_subset, class_set(binary_pattern))) = 0;
    
    tmp_hier.b_isTree = 1;
    tmp_hier.Root = [];
    tmp_hier.node_id = 1;
    tmp_hier.branch_link = cell(2, 1);
    tmp_hier.branch_class{1, 1} = class_set(binary_pattern);
    tmp_hier.branch_class{2, 1} = class_set(~binary_pattern);

    fprintf('processing node %d %d/%d\n', node_id, i, upper_size);
    result = classify_crossValidation(feature_subset, featurei, classid_subset, traj, fold, classifier, traj_vote, tmp_hier, test_set, cv_one_fold);
    if result.cv_class_recall > max_score
        max_score = result.cv_class_recall;
        max_pattern = binary_pattern;
    end
end

if max_score < 0
    return;
end

hier.branch_class{1, 1}= class_set(max_pattern);
hier.branch_class{2, 1}= class_set(~max_pattern);

hier.branch_link{1, 1}= append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, hier.branch_class{1, 1}, node_id + 1);
hier.branch_link{2, 1}= append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, hier.branch_class{2, 1}, node_id + 2);

end

