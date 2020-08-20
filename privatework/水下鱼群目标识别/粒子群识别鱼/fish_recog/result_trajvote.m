function [ predict_vote, predict_vote_single ] = result_trajvote( predict_src, traj, score, ignore_minus )

if ~exist('ignore_minus','var'), ignore_minus = 0;end

sample_num = length(predict_src);
predict_vote = zeros(sample_num, 1);

traj_id = unique(traj);
traj_num = length(traj_id);

class_id = unique(predict_src);

if ~exist('score', 'var')
    score  = zeros(length(predict_src), 1);
end
%class_range = min(predict_src):max(predict_src);
predict_vote_single = zeros(traj_num, 1);
for i = 1:traj_num
    traj_indic = traj == traj_id(i);
    tmp_predict = predict_src(traj_indic);
    tmp_score = score(traj_indic);
    
    tmp_distrib = zeros(length(class_id), 1);
    for j = 1:length(tmp_predict)
        if ignore_minus && tmp_predict(j) == 0; continue; end
        find_index = find(class_id==tmp_predict(j));
        tmp_distrib(find_index) = tmp_distrib(find_index) + 1 + tmp_score(j)/10000;
    end
    
    [mv, mi] = max(tmp_distrib);
    predict_vote(traj_indic)=class_id(mi);
    predict_vote_single(i)=class_id(mi);
end

end