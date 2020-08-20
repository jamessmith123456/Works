function [ result ] = result_cv_evaluate( cv_result )
%RESULT_CV_EVALUATE Summary of this function goes here
%   Detailed explanation goes here
cv_fold = length(cv_result);
for i = 1:cv_fold
    result.detail_recall(i,:) = cv_result{i}.classrecall;
    result.detail_precision(i,:) = cv_result{i}.classprecision;
    
    result.cv_recall(i,1) = cv_result{i}.recallAver;
    result.cv_precision(i,1) = cv_result{i}.precisionAver;
    result.cv_count(i,1)= cv_result{i}.recallCount;
end

result.aver_recall = mean(result.cv_recall);
result.aver_precision = mean(result.cv_precision);
result.aver_count = mean(result.cv_count);

result.class_recall = mean(result.detail_recall);
result.class_precision = mean(result.detail_precision);

result.cv_result = cv_result;
end

