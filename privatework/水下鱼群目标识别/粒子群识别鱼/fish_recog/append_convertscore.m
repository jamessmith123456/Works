function [ score_convert ] = append_convertscore( convert_table, score )

if iscell(score)
    score = cell2mat(score);
end

[samples, class_num] = size(score);
score_convert = zeros(samples, class_num);
for i = 1:class_num
    score_convert(:,i) = score(:, convert_table(i, 3));
end

end

