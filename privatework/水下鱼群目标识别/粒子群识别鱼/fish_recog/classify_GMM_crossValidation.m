function [  result] = classify_GMM_crossValidation(features, classid, valid_class, cv_id, test_set)

mute = 1;

if ~exist('test_set','var'), test_set = 1;end

if 0 ~= test_set && 1 ~= test_set
    test_set = 1;
end
unused_set = 1 - test_set;

%if cv_id is a number, then it is the fold. generate cv_id.
if length(cv_id) == 1
    fold = cv_id;
    traj_id = unique(traj);
    traj_num = length(traj_id);
    traj_class = zeros(traj_num, 1);
    for i = 1:traj_num
        temp_class = classid(traj==traj_id(i));
        traj_class(i) = temp_class(1);
    end
    rand('state', 1);
    traj_indices = crossvalind('Kfold',traj_class, fold);
    
    cv_id = zeros(length(traj), 1);
    for i = 1:fold
        cv_id(ismember(traj, traj_id(traj_indices==i))) = i;
    end
end

cv_id_set = unique(cv_id);
fold = length(cv_id_set);

for i = 1:fold
    tmp_test_id = mod(i+test_set-1, fold)+1;
    tmp_unused_id = mod(i+unused_set-1, fold)+1;
    
    fish_test = (cv_id==cv_id_set(tmp_test_id));
    fish_unused = (cv_id==cv_id_set(tmp_unused_id));
    fish_train = ~fish_test & ~fish_unused;
    
    fish_train = fish_train & classid == valid_class;
    fish_test_valid = fish_test & classid == valid_class;
    fish_test_invalid = fish_test & classid ~= valid_class;
    
    if ~mute
        fprintf('%s Prcessing fold %d/%d\t train: %d\ttest_valid: %d\ttest_invalid: %d\n', append_timeString(), i, fold, sum(fish_train), sum(fish_test_valid), sum(fish_test_invalid));
    end
    
    [result.liklihood_valid{i,1}, result.liklihood_invalid{i,1}, result.model(i,1)] = classify_GMM( features(fish_train,:), features(fish_test_valid,:), features(fish_test_invalid,:) );
    result.aver_liklihoodValid(i,1) = mean(result.liklihood_valid{i,1});
    result.aver_liklihoodInvalid(i,1) = mean(result.liklihood_invalid{i,1});
    result.distance(i,1) = result.aver_liklihoodValid(i,1) - result.aver_liklihoodInvalid(i,1);
    
    if ~mute
        fprintf('%s Prcessed \t validMean: %f\tinValidMean: %f\tDiffer: %f\n', append_timeString(), result.aver_liklihoodValid(i,1), result.aver_liklihoodInvalid(i,1), result.distance(i,1));
    end
end

result.averDistance = mean(result.distance);
end