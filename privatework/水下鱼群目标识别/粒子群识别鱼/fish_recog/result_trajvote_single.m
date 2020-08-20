function [ predict_vote ] = result_trajvote_single( predict_src)

sample_num = length(predict_src);
predict_vote = zeros(sample_num, 1);

class_id = unique(predict_src);

tmp_distrib = zeros(length(class_id), 1);
for j = 1:length(predict_src)
    find_index = find(class_id==predict_src(j));
    tmp_distrib(find_index) = tmp_distrib(find_index) + 1;
end

[mv, mi] = max(tmp_distrib);
predict_vote(:)=class_id(mi);

end