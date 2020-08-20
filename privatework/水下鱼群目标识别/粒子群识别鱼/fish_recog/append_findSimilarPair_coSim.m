function [ pairs ] = append_findSimilarPair_coSim( confus )

pairs = {};
[class_num, class_num2] = size(confus);
if class_num ~= class_num2
    return;
end

normal = repmat(sum(confus, 2), 1, class_num);
confus = confus./normal;

co_confus = zeros(class_num, class_num);
for i = 1:class_num
    for j = i+1:class_num
        co_confus(i,j)=confus(i,j)+confus(j,i);
    end
end

for i = 1:class_num/2
    if sum(sum(co_confus)) == 0
        break;
    end
    
    [sort_score, sort_index] = max(co_confus);
    [sort2_score,sort2_index]= max(sort_score);
    pair_1=sort2_index;
    pair_2=sort_index(sort2_index);
    
    co_confus(pair_1,:)=0;
    co_confus(:,pair_1)=0;
    
    co_confus(pair_2,:)=0;
    co_confus(:,pair_2)=0;
    
    pairs{end+1,1}=[pair_1, pair_2];
    pairs{end,2}=sort2_score;
end

end

