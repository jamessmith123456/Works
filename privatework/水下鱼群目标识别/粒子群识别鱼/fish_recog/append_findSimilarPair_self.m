function [ pairs ] = append_findSimilarPair( confus )

pairs = {};
[class_num, class_num2] = size(confus);
if class_num ~= class_num2
    return;
end

normal = repmat(sum(confus, 2), 1, class_num);
confus = confus./normal;
for i = 1:class_num/2
    tmp_confus = confus;
    tmp_confus = tmp_confus./repmat(diag(tmp_confus)+eps, 1, class_num);
    tmp_confus(1:class_num+1:end) = 0;
    
    if sum(sum(tmp_confus)) == 0
        break;
    end
    
    [sort_score, sort_index] = sort(tmp_confus, 2, 'descend');
    [sort2_score,sort2_index]=sort(sort_score, 'descend');
    pair_1=sort2_index(1);
    pair_2=sort_index(pair_1,1);
    
    pairs{end+1,1}=[pair_1, pair_2];
    pairs{end,2}=[sort2_score(1, 1), confus(pair_1, pair_2), confus(pair_1, pair_1)];
    
    confus(pair_1,:)=0;
    confus(:,pair_1)=0;
    confus(pair_1,pair_1)=1;
    
    confus(pair_2,:)=0;
    confus(:,pair_2)=0;
    confus(pair_2,pair_2)=1;
end

end

