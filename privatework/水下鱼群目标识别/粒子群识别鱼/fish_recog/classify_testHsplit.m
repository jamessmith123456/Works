function [ scores ] = classify_testHsplit( feature, featurei, class, traj, class_valid )

classifier = 2;
cvfold = 6;

Hier.branch_link = cell(2,1);
Hier.branch_class = {};

class_id = unique(class);
class_num = length(class_id);
class_indicate = ismember(class_id, class_valid);
class_valids = class_id(class_indicate);
class_invalid = class_id(~class_indicate);

valide_num = sum(class_indicate);
test_num = 2^(valide_num-1);

scores = zeros(test_num, 3+class_num);
for i = 1:test_num-1
    tmp_hier = Hier;
    
    tmp_valid_pattern = dec2bin(i,valide_num) == '1';    
    tmp_whole_pattern = -1 * ones(1, class_num);
    tmp_whole_pattern(class_indicate) = tmp_valid_pattern;

    scores(i,1) = i;
    scores(i,3) = min(sum(tmp_valid_pattern), valide_num-sum(tmp_valid_pattern));
    scores(i,4:end)=tmp_whole_pattern;

    tmp_hier.branch_class{1} = class_valids(tmp_valid_pattern);
    tmp_hier.branch_class{2} = class_valids(~tmp_valid_pattern);

    [ count_recall, class_precision, scores(i,2)] = classify_crossValidation_tmp(feature, featurei, class, traj, cvfold, classifier, tmp_hier);
end

scores(end,1) = 0;
scores(end,3) = 1;
tmp_pattern = class_id;
tmp_pattern(~class_indicate)=-1;
scores(end,4:end)=tmp_pattern;

% tmp_class = class;
% tmp_class(ismember(tmp_class, class_invalid))=-1;
% [ count_recall, class_precision, scores(end,2)] = classify_crossValidation(feature, featurei, tmp_class, traj, 6, 3);
indHier.branch_link = cell(valide_num,1);
indHier.branch_class = cell(valide_num,1);
for i = 1:valide_num
    indHier.branch_class{i}=class_valids(i);
end
[ count_recall, class_precision, scores(end,2)] = classify_crossValidation_tmp(feature, featurei, class, traj, cvfold, classifier, indHier);

end

