function [featureSubset, scoreResult, more_info] = classify_featureselection_fbtrain(features, feature_id, classid, traj, initfeatureSubset, standard, fold, iter, savefile)

feature_hist = unique(feature_id);
featuresnum = length(feature_hist);

% tmpval = zeros(featuresnum, 1);
% for i = 1:featuresnum
%     tmpid = feature_hist(i);
%     tmpval(i) = getEvalueScore(tmpid, features, feature_id, classid, traj, fold, standard);
% end
% [drop, sortindex] = sort(tmpval, 'descend');
% 
% init_num = 20;
% featureSubset = feature_hist(sortindex(1:init_num));
featureSubset=initfeatureSubset;%[32,33,35,34,36,37,40,38,39,41,50,52,48,49,51,42,13,27,30,28];

scoreResult = [];
featureUsed = zeros(featuresnum, 1);
%featureUsed(ismember(feature_hist, featureSubset))=1;

factor_filter = 0.02;
factor_addin = 0.02;

more_info.actions = zeros(iter, 9);
more_info.initfeatureSubset = cell(iter, 1);
more_info.resultfeatureSubset = cell(iter, 1);

for loop1 = 1:iter
    fprintf('\nBegin %d/%d\n', loop1, iter);
    more_info.initfeatureSubset{loop1} = featureSubset;

    %remove the worst feature
    subfeature_size = length(featureSubset);
    if 0 < subfeature_size
        tmpval = zeros(subfeature_size,1);
        for loop2_i = 1:subfeature_size
            tmpfeatureSubset = featureSubset;
            tmpfeatureSubset(loop2_i) = [];
            
            tmpval(loop2_i) = getEvalueScore(tmpfeatureSubset, features, feature_id, classid, traj, fold, standard);
        end
        
        tmpval_imp = tmpval - factor_filter * featureUsed(featureSubset);
        
        [maxv_filter, filtered_index] = max(tmpval_imp);
        filtered_id = featureSubset(filtered_index);
        maxv_filter_ori = tmpval(filtered_index);
    else
        maxv_filter = -1000;
        filtered_id = -1;
        maxv_filter_ori = 0;
    end
    
    %add the best feature
    candidate_id = feature_hist(~ismember(feature_hist, featureSubset));
    candidate_num = length(candidate_id);
    if 0 < candidate_num
        tmpval = zeros(candidate_num,1);
        for loop2_i = 1:candidate_num
            tmpfeatureSubset = featureSubset;
            tmpfeatureSubset(end+1) = candidate_id(loop2_i);
            
            tmpval(loop2_i) = getEvalueScore(tmpfeatureSubset, features, feature_id, classid, traj, fold, standard);
        end
        tmpval_imp = tmpval - factor_addin * featureUsed(candidate_id);
        
        [maxv_addin, bestIndex] = max(tmpval_imp);
        addin_id = candidate_id(bestIndex);
        maxv_addin_ori = tmpval(bestIndex);
    else
        addin_id = -1;
        maxv_addin = -1000;
        maxv_addin_ori = 0;
    end
    
    more_info.actions(loop1, 1) = filtered_id;
    if 0 < filtered_id
        more_info.actions(loop1, 2) = featureUsed(filtered_id);
    end
    more_info.actions(loop1, 3) = maxv_filter;
    more_info.actions(loop1, 4) = maxv_filter_ori;
    
    more_info.actions(loop1, 5) = addin_id;
    if 0 < addin_id
        more_info.actions(loop1, 6) = featureUsed(addin_id);
    end
    more_info.actions(loop1, 7) = maxv_addin;
    more_info.actions(loop1, 8) = maxv_addin_ori;

    if maxv_addin > maxv_filter
    	featureSubset(end+1) = addin_id;
    	featureUsed(addin_id) = featureUsed(addin_id) + 1;
        more_info.actions(loop1, 9) = 1;
    else
    	featureSubset(filtered_index) = [];
        more_info.actions(loop1, 9) = 2;
    end

    
    more_info.resultfeatureSubset{loop1} = featureSubset;
    
save(savefile, 'more_info');
end

end

function score = getEvalueScore(featureSubset, features, feature_id, classid, traj, fold, standard)

subset_index = ismember(feature_id, featureSubset);
subset_feature = features(:, subset_index);

[ scores(1,1), scores(1,2), scores(1,3)] = classify_crossValidation(subset_feature, classid, traj, fold);

score = scores(1, standard);


end