function [ result_score ] = append_hierScoreToFlat( hierTree, hierScore )
%APPEND_HIERSCORETOFLAT Summary of this function goes here

[samples, levels] = size(hierScore);
result_score = zeros(samples, max(hierTree.Input_classid_set));

for i = 1:samples
    score_depth = inner_scoreDepth(hierScore(i, :));
    [result_score(i,:)] = inner_addNodeScore(hierTree, hierScore(i,:), result_score(i,:), 1) / score_depth;
end

result_score = mat2cell(result_score, ones(size(result_score, 1),1), size(result_score, 2));
end

function score_depth = inner_scoreDepth(score)
    score_depth = 1;
    for i = 2:length(score)
        if isempty(score{i})
            break;
        end
        score_depth = i;
    end
end

function [result_score] = inner_addNodeScore(hierTree, hierScore, result_score, levels)
if levels > length(hierScore) || isempty(hierScore{levels})
    return;
end

for i = 1:length(hierScore{levels})
    node_predict = hierTree.model.classid_array(i);
    if -1 == node_predict
        continue;
    end
    
    for j = 1:length(hierTree.branch_class{node_predict})
        result_score(hierTree.branch_class{node_predict}(j)) = result_score(hierTree.branch_class{node_predict}(j)) + hierScore{levels}(i);
    end
end

[mv, mi] = max(hierScore{levels});
node_predict = hierTree.model.classid_array(mi);
if -1 ~= node_predict
    [result_score] = inner_addNodeScore(hierTree.branch_link{node_predict}, hierScore, result_score, levels+1);
end

end