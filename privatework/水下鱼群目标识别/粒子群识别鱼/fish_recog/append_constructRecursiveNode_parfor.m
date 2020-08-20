function [ hier ] = append_constructRecursiveNode_parfor( feature, featurei, classid, traj, fold, classifier, traj_vote, class_set, node_id, cv_one_fold)
%APPEND_CONSTRUCTRECURSIVENODE Summary of this function goes here

if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

class_size = length(class_set);
if class_size <= 4
    return;
end

hier.node_id = node_id;
hier.class_id = classid;

class_size_left = floor(class_size / 2);

class_subset_index = ismember(classid, class_set);
%sample_subset = sum(class_subset_index);
feature_subset = feature(class_subset_index, :);
classid_subset = classid(class_subset_index);

upper_size = 2 ^ class_size;
%max_score = -0.1;
%max_pattern = 0;
score_set = zeros(upper_size, 1);
pattern_set = zeros(upper_size, 1);

if ~matlabpool('size')
    matlabpool('open');
end
tmp_hier(upper_size).a = 1;
parfor i = 1:upper_size-1
    binary_pattern = de2bi(i, class_size) == 1;
    if class_size_left ~= sum(binary_pattern)
        continue;
    end
    
%     tmp_classid = ones(sample_subset, 1);
%     tmp_classid(ismember(classid_subset, class_set(binary_pattern))) = 0;
    
    tmp_hier(i).b_isTree = 1;
    tmp_hier(i).Root = [];
    tmp_hier(i).branch_link = cell(2, 1);
    tmp_hier(i).branch_class{1, 1} = class_set(binary_pattern);
    tmp_hier(i).branch_class{2, 1} = class_set(~binary_pattern);

    
    result = classify_crossValidation(feature_subset, featurei, classid_subset, traj, fold, classifier, traj_vote, tmp_hier(i), cv_one_fold, 0);
    score_set(i) = result.cv_class_recall;
    pattern_set(i) = binary_pattern;
end
if matlabpool('size')
    matlabpool('close');
end

[max_score, max_index] = max(score_set);
max_pattern = pattern_set(max_index);
if max_score <= 0
    return;
end

hier.branch_class{1, 1}= class_set(max_pattern);
hier.branch_class{2, 1}= class_set(~max_pattern);

save(sprintf('part_%02dlmat',node_id),  hier);

hier.branch_link{1, 1}= append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, hier.branch_class{1, 1}, node_id + 1);
hier.branch_link{2, 1}= append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, hier.branch_class{2, 1}, node_id + 2);

end

