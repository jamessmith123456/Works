function [predict_output, score_output] = result_1vs1( classifier_number, score_input, class_id, class_weight )

%convert 1vs1 voting score to 1vs. rest score.
[sample_number, dimens] = size(score_input);

predict_output = [];
vote_output = [];
if classifier_number * (classifier_number - 1) ~= 2 * dimens
    return;
end

converter_postive = [];
converter_negtive = [];

%find voting item positive/negetive indicator
for i = 1:classifier_number-1
    converter_postive = [converter_postive, ones(1, classifier_number-i)*i];
    converter_negtive = [converter_negtive, i+1:classifier_number];
end

vote_output = zeros(sample_number, classifier_number);
confident_output = zeros(sample_number, classifier_number);

score_input_constraint = score_input;
%score_input_constraint(score_input_constraint<-1)=-1;
%score_input_constraint(score_input_constraint>1)=1;

for i = 1:sample_number
    %begin voting
    tmp_indicates_postive = score_input(i, :)>0;
    tmp_indicates_negtive = score_input(i, :)<0;
    
    suc_positive = converter_postive(tmp_indicates_postive);
    suc_negitive = converter_negtive(tmp_indicates_negtive);
    
    if 0 < sum(suc_positive)
        vote_output(i, :) = vote_output(i, :) + hist(suc_positive, 1:classifier_number);
    end
    
    if 0 < sum(suc_negitive)
        vote_output(i, :) = vote_output(i, :) + hist(suc_negitive, 1:classifier_number);
    end
    %end voting
    
    for j = 1:classifier_number
        confident_output(i,j)=sum(score_input_constraint(i,converter_postive==j))+ (-1)*sum(score_input_constraint(i,converter_negtive==j));
    end
end

score_output = vote_output / max(1, (classifier_number-1));
confident_output = confident_output / max(1, (classifier_number-1));

weight_cof = 100;

if exist('class_weight','var') 
    add_weight = reshape(class_weight, 1, classifier_number);
    add_weight = repmat(add_weight, sample_number, 1);
    score_voting = score_output + add_weight/weight_cof;
else
    score_voting = score_output + confident_output/weight_cof;
end

[mv, predict_vote] = max(score_voting, [], 2);
predict_output=class_id(predict_vote);

end