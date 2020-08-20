function [ BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23 ] = interface_classification( features, models )

[BGOT_result_predict, BGOT_result_scores_23] = classify_HierSVM_predict(features, models);

if iscell(BGOT_result_scores_23)
    BGOT_result_scores_23 = cell2mat(BGOT_result_scores_23);
end
BGOT_result_score = zeros(length(BGOT_result_predict), 1);
for i = 1:length(BGOT_result_predict)
    BGOT_result_score(i,1) = BGOT_result_scores_23(i, BGOT_result_predict(i));
end
end

