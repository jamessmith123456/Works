function [ features, RTN_classid, traj_id, cv_id ] = append_filteClassGroup( new_classGroup, features, class_id, traj_id, cv_id, boolMinus1 )

if ~exist('boolMinus1', 'var')
    boolMinus1 = 0;
end

allClass = [];
for i = 1:length(new_classGroup)
    allClass = [allClass, new_classGroup{i}];
end

if boolMinus1
    minusClass = setdiff(unique(class_id), allClass);
    allClass = [allClass, minusClass];
end

indic = ismember(class_id, allClass);
class_id = class_id(indic);
features = features(indic,:);
traj_id = traj_id(indic);
cv_id = cv_id(indic);

RTN_classid = -1 * ones(sum(indic), 1);
for i = 1:length(new_classGroup)
    indic = ismember(class_id, new_classGroup{i});
    RTN_classid(indic)=i;
end


end

