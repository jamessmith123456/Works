function [ scores ] = append_convertScore_allNodes_15( score_allNodes )
treePath = {
    {[1,2,3,4,7,8,9,11];[5,6,10,12,13,14,15]};
    {[1,7,9,11];[2,3,4,8];};
    {1;7;9;11;};
    {2;3;4;8;};
    {[5,10,15];[6,12,13,14];};
    {5;10;15;};
    {6;12;13;14;};
    };
    

sample_num = length(score_allNodes{1,2});
class_num = 15;
scores = ones(sample_num, class_num);
for i = 1:sample_num
    for j = 1:length(treePath)
        for m = 1:length(treePath{j})
            for n = 1:length(treePath{j}{m})
            	scores(i,treePath{j}{m}(n))=scores(i,treePath{j}{m}(n))*score_allNodes{j,2}{i}(m);
            end
        end
    end
end

end

