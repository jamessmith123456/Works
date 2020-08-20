function [ result_traj_predict, result_traj_scores, result_traj_id ] = result_trajvote_byScore( score, traj_id )

result_traj_predict = {};
result_traj_scores = {};
result_traj_id = unique(traj_id);
traj_num = length(result_traj_id);
if size(score, 1) ~= length(traj_id)
    return;
end

result_traj_predict = cell(traj_num, 1);
result_traj_scores = cell(traj_num, 1);
for i = 1:traj_num
    traj_indic = traj_id == result_traj_id(i);
    tmp_score = score(traj_indic,:);
    if sum(traj_indic) > 1
        tmp_score = median(tmp_score);
    end
    
    [mv1, mi1] = max(tmp_score);
    if 0 == mv1
        result_traj_predict{i}=0;
        result_traj_scores{i}=0;
        continue;
    end
    tmp_score(mi1) = 0;
    [mv2, mi2] = max(tmp_score);
    if mv2 > 0
        mv1 = [mv1; mv2];
        mi1 = [mi1; mi2];
        tmp_score(mi2) = 0;
        [mv3, mi3] = max(tmp_score);
        if mv3 > 0
            mv1 = [mv1; mv3];
            mi1 = [mi1; mi3];
        end
    end
    result_traj_predict{i}=mi1;
    result_traj_scores{i}=mv1;
end

end