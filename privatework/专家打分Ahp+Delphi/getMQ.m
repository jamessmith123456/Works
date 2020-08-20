function [result] = getMQ(score,e)
%score是6位专家对5个指标的打分 5行6列
result = zeros(size(score,1),5);
for i=1:size(score,1)
    temp_score = score(i,:);
    temp_score = sort(temp_score);
    temp_M = (temp_score(3)+temp_score(4))/2; %因为是6位专家 所以是第7个
    temp_Q1 = 0.75*temp_score(2)+0.25*temp_score(1);
    temp_Q3 = 0.75*temp_score(5)+0.25*temp_score(6);
    temp_Q31 = temp_Q3-temp_Q1;
    result(i,1) = temp_M;
    result(i,2) = temp_Q1;
    result(i,3) = temp_Q3;
    result(i,4) = temp_Q31;
    if temp_Q31<=e
        result(i,5) = 0;
    else
        result(i,5) = 1;
    end
end

end

