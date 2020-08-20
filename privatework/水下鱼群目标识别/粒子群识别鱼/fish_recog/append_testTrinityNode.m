function [ result ] = append_testTrinityNode( features, featurei, class_id, cv_id, traj_id,  tripleClassSet)

result = [];

indic = ismember(class_id, tripleClassSet);
class_id = class_id(indic);
features = features(indic, :);
cv_id = cv_id(indic);
traj_id = traj_id(indic);

tmp_id = class_id;
for i = 1:length(tripleClassSet)
    class_id(tmp_id == tripleClassSet(i)) = i;
end

for i = 1:length(tripleClassSet)
    tmp_id = class_id;
    tmp_indic = tmp_id ~= i;
    
    tmp_id(tmp_indic) = 0;
    [  tmp_result1] = classify_crossValidation_byID(features, featurei, tmp_id, traj_id, cv_id, 2, 0, 0, 0);
    
    [  tmp_result2] = classify_crossValidation_byID(features(tmp_indic,:), featurei, class_id(tmp_indic), traj_id(tmp_indic), cv_id(tmp_indic), 2, 0, 0, 0);

    cv_fold = length(unique(cv_id));
    result(i,1) = mean([tmp_result1.cv_class_recall_all(cv_fold+1,1) * tmp_result2.cv_class_recall, tmp_result1.cv_class_recall_all(cv_fold+1,2)]);
    result(i,2) = tmp_result1.cv_class_recall;
    result(i,3) = tmp_result2.cv_class_recall;
end


[  tmp_result] = classify_crossValidation_byID(features, featurei, class_id, traj_id, cv_id, 2, 0, 0, 0);
result(end+1,1) = tmp_result.cv_class_recall;

end


