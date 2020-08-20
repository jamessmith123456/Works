function [featureSubset, scoreResult] = classify_featureselection_pool(features, feature_id, classid, traj, maxfeature, standard, fold, classifier, filesavename, traj_vote, hier_tree, cv_one_fold, poolSize)

if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

if ~exist('poolSize','var')
    poolSize = maxfeature * 3;
end

if ~exist('traj_vote','var')
    traj_vote = 0;
end

if ~exist('filesavename','var')
    filesavename = 'featureselection.mat';
end

%use validatation set as test set
test_set = 0;

if ~exist('hier_tree','var')  || hier_tree == 0
    clear hier_tree;
    hier_tree.invalid = 1;
end

if ~exist('classifier','var')
    classifier = 2;
end

if ~exist('fold','var')
    fold = 6;
end

if ~exist('standard','var')
    standard = 3;
end

feature_hist = unique(feature_id);
featuresnum = length(feature_hist);
if exist('maxfeature','var')
    maxfeature = featuresnum;
end
maxfeature = min(maxfeature, featuresnum);
poolSize = min(poolSize, featuresnum);

featureSubset = [];
loopstart = 1;
scoreResult = zeros(maxfeature, 3);

if exist([filesavename,'.mat'],'file')
    load(filesavename);

%     scoreResult = tmp_data.scoreResult;
%     featureSubset = tmp_data.featureSubset;
%     featurePool = tmp_data.featurePool;
%     indiv_performance = tmp_data.indiv_performance;
    loopstart = length(featureSubset) + 1;

    ctime = clock();
    old_filesavename = sprintf('%s_%d_%d_%d_%d_%d_old.mat', filesavename, ctime(1), ctime(2), ctime(3), ctime(4), ctime(5));
    movefile([filesavename,'.mat'], old_filesavename);
else
    indiv_performance = zeros(featuresnum, 3);
    for loop_i = 1:featuresnum
        loop = feature_hist(loop_i);
        
        subset_index = ismember(feature_id, loop);
        subset_feature = features(:, subset_index);
        subset_id = feature_id(subset_index);
        
        result = classify_crossValidation(subset_feature, subset_id, classid, traj, fold, classifier, traj_vote, hier_tree, cv_one_fold, test_set);
        indiv_performance(loop_i, 1) = result.cv_count_recall;
        indiv_performance(loop_i, 2) = result.cv_class_precision;
        indiv_performance(loop_i, 3) = result.cv_class_recall;
        
        fprintf('poolSelect: %d/%d id:%d count_recall:%f class_precision:%f class_recall:%f\n', loop_i, featuresnum, loop, indiv_performance(loop_i, 1), indiv_performance(loop_i, 2), indiv_performance(loop_i, 3));
    end
    
    [sort_v, sort_i] = sort(indiv_performance(:,standard), 'descend');
    featurePool = sort_i(1:poolSize);
    
    save(filesavename, 'featureSubset', 'scoreResult', 'featurePool', 'indiv_performance');
end

for loop1 = loopstart:maxfeature
    fprintf('\nBegin %d/%d\n', loop1, maxfeature);

    tmpFeatureSubset = zeros(featuresnum,length(featureSubset)+1);
    tmpval = zeros(featuresnum,3);
    
    for loop2_i = 1:length(featurePool);
        loop2 = featurePool(loop2_i);
        if(find(featureSubset==loop2))
            continue;
        end
        
        for i = 1:length(featureSubset)
            tmpFeatureSubset(loop2_i,i)=featureSubset(i);
        end
        tmpFeatureSubset(loop2_i,end) = loop2;
        
        subset_index = ismember(feature_id, tmpFeatureSubset(loop2_i,:));
        subset_feature = features(:, subset_index);
        subset_id = feature_id(subset_index);
        
        result = classify_crossValidation(subset_feature, subset_id, classid, traj, fold, classifier, traj_vote, hier_tree, cv_one_fold, test_set);
        tmpval(loop2_i, 1) = result.cv_count_recall;
        tmpval(loop2_i, 2) = result.cv_class_precision;
        tmpval(loop2_i, 3) = result.cv_class_recall;
            
        fprintf('select: %d/%d test %d/%d id:%d count_recall:%f class_precision:%f class_recall:%f\n', loop1, maxfeature, loop2_i, length(featurePool), loop2, tmpval(loop2_i, 1), tmpval(loop2_i, 2), tmpval(loop2_i, 3));
    end
    [currentbest, bestIndex] = max(tmpval(:,standard));
    
    featureSubset = tmpFeatureSubset(bestIndex,:);
    scoreResult(loop1,:) = tmpval(bestIndex, :);

    fprintf('Choosed %d\n Score: %f count_recall:%f class_precision:%f class_recall:%f\n', bestIndex, currentbest, tmpval(bestIndex, 1), tmpval(bestIndex, 2), tmpval(bestIndex, 3));
    fprintf('Max value: count_recall:%f class_precision:%f class_recall:%f\n', max(tmpval(:,1)), max(tmpval(:,2)), max(tmpval(:,3)));

    save(filesavename, 'featureSubset', 'scoreResult', 'featurePool', '-append');
end

end