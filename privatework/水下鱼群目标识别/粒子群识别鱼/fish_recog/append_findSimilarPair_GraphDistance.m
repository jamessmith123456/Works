function [ pairs ] = append_findSimilarPair_GraphDistance( confus )

pairs = {};
[class_num, class_num2] = size(confus);
if class_num ~= class_num2
    return;
end

normal = repmat(sum(confus, 2), 1, class_num);
confus = confus./normal;

distance = zeros(class_num, class_num);
indics = ones(class_num, 1);
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

for i = 1:class_num/2
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

pairs{end+1,1}=find(indics);
end

