function [ pairs ] = append_findSimilarPair_top2( confus )

pairs = {};
[class_num, class_num2] = size(confus);
if class_num ~= class_num2
    return;
end

normal = repmat(sum(confus, 2), 1, class_num);
confus = confus./normal;

distance = zeros(class_num, class_num);
for i = 1:class_num
    for j = i+1:class_num
        distance(i,j) = confus(i,j)+confus(j,i);
        
        for m = 1:class_num
            if m == i || m == j
                continue;
            end
            
            distance(i,j) = distance(i,j) + confus(i,m)*confus(m,j) + confus(j,m)*confus(m,i);
        end
    end
end

indics = ones(class_num, 1);
for i = 1:class_num/2
    tmp_confus = confus;
    tmp_confus = tmp_confus./repmat(diag(tmp_confus)+eps, 1, class_num);
    tmp_confus(1:class_num+1:end) = 0;
    
    if sum(sum(tmp_confus)) == 0
        break;
    end
    
    [sort_score, sort_index] = sort(confus, 2, 'descend');
    [sort2_score,sort2_index]=sort(sort_score(:,2)./(sort_score(:,1)+eps), 'descend');
    pair_1=sort_index(sort2_index(1),1);
    pair_2=sort_index(sort2_index(1),2);
    
    pairs{end+1,1}=[pair_1, pair_2];
    pairs{end,2}=[sort2_score(1, 1), confus(sort2_index(1), pair_1), confus(sort2_index(1), pair_2)];
    
    confus(pair_1,:)=0;
    confus(:,pair_1)=0;
    
    confus(pair_2,:)=0;
    confus(:,pair_2)=0;
    
    indics(pair_1) = 0;
    indics(pair_2) = 0;
end

for i = 1:class_num
    if ~indics(i)
        distance(i,:)=0;
        distance(:,i)=0;
    end
end

for i = 1:class_num
    if sum(sum(distance)) == 0
        break;
    end
    
    [sort_score, sort_index] = sort(distance, 2, 'descend');
    [sort2_score,sort2_index]=sort(sort_score(:,1), 'descend');
    pair_1=sort2_index(1);
    pair_2=sort_index(pair_1,1);
    
    pairs{end+1,1}=[pair_1, pair_2];
    pairs{end,2}=[sort2_score(1, 1)];
    
    distance(pair_1,:)=0;
    distance(:,pair_1)=0;
    
    distance(pair_2,:)=0;
    distance(:,pair_2)=0;
    
    indics(pair_1) = 0;
    indics(pair_2) = 0;
end

end

