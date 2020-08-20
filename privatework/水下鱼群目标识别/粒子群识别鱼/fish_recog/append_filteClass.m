function [ features, class_id, traj_id, cv_id ] = append_filteClass( new_classSet, features, class_id, traj_id, cv_id )

indic = ismember(class_id, new_classSet);
class_id = class_id(indic);
features = features(indic,:);
traj_id = traj_id(indic);
cv_id = cv_id(indic);

for i = 1:length(new_classSet)
class_id(class_id==new_classSet(i))=i;
end


end

