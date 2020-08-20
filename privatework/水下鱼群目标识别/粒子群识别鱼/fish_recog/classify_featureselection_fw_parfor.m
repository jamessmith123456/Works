function [featureSubset, scoreResult] = classify_featureselection_fw_parfor(features, feature_id, classid, traj, maxfeature, standard, fold, classifier, filesavename, traj_vote, hier_tree, cv_one_fold, cv_id)

if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

if ~exist('cv_id','var')
    cv_id = [];
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

featureSubset = [];
loopstart = 1;
scoreResult = zeros(maxfeature, 3);

if exist([filesavename,'.mat'],'file')
    tmp_data = load(filesavename);
    ctime = clock();
    old_filesavename = sprintf('%s_%d_%d_%d_%d_%d_old.mat', filesavename, ctime(1), ctime(2), ctime(3), ctime(4), ctime(5));
    movefile([filesavename,'.mat'], old_filesavename);

    scoreResult = tmp_data.scoreResult;
    featureSubset = tmp_data.featureSubset;
    loopstart = length(featureSubset) + 1;
end

if ~matlabpool('size')
    matlabpool('open');
end
baseFile = ('part_%04d.mat');
for loop1 = loopstart:maxfeature
    fprintf('\nBegin %d/%d\n', loop1, maxfeature);

    subsetsize = length(featureSubset)+1;
    tmpFeatureSubset = zeros(featuresnum, subsetsize);
    tmpval = zeros(featuresnum,3);
    
    subset_feature = cell(featuresnum, 1);
    subset_id = cell(featuresnum, 1);
    for loop2_i = 1:featuresnum
        loop2 = feature_hist(loop2_i);
        if(find(featureSubset==loop2))
            continue;
        end
        
        for i = 1:length(featureSubset)
            tmpFeatureSubset(loop2_i,i)=featureSubset(i);
        end
        tmpFeatureSubset(loop2_i,subsetsize) = loop2;
        
        subset_index = ismember(feature_id, tmpFeatureSubset(loop2_i,:));
        subset_feature{loop2_i,1} = features(:, subset_index);
        subset_id{loop2_i,1} = feature_id(subset_index);
    end
    save(sprintf(baseFile, loop1), 'tmpval', 'tmpFeatureSubset');
    
    parfor loop2_i = 1:featuresnum
        loop2 = feature_hist(loop2_i);
        if(find(featureSubset==loop2))
            continue;
        end
        
        result = classify_crossValidation(subset_feature{loop2_i,1}, subset_id{loop2_i,1}, classid, traj, fold, classifier, traj_vote, hier_tree, cv_one_fold, test_set);
        saveToFile(sprintf(baseFile, loop1), result, loop2_i);
            
        fprintf('select: %d/%d test %d/%d id:%d count_recall:%f class_precision:%f class_recall:%f\n', loop1, maxfeature, loop2_i, featuresnum, loop2, result.cv_count_recall, result.cv_class_precision, result.cv_class_recall);
    end
    load(sprintf(baseFile, loop1), 'tmpval', 'tmpFeatureSubset');
    [currentbest, bestIndex] = max(tmpval(:,standard));
    
    featureSubset = tmpFeatureSubset(bestIndex,:);
    scoreResult(loop1,:) = tmpval(bestIndex, :);

    fprintf('Choosed %d\n Score: %f count_recall:%f class_precision:%f class_recall:%f\n', bestIndex, currentbest, tmpval(bestIndex, 1), tmpval(bestIndex, 2), tmpval(bestIndex, 3));
    fprintf('Max value: count_recall:%f class_precision:%f class_recall:%f\n', max(tmpval(:,1)), max(tmpval(:,2)), max(tmpval(:,3)));

    save(filesavename, 'featureSubset', 'scoreResult');
end
if matlabpool('size')
    matlabpool('close');
end

end

function saveToFile(fileName, result, loop2_i)
load(fileName, 'tmpval', 'tmpFeatureSubset');
tmpval(loop2_i, 1) = result.cv_count_recall;
tmpval(loop2_i, 2) = result.cv_class_precision;
tmpval(loop2_i, 3) = result.cv_class_recall;
save(fileName, 'tmpval', 'tmpFeatureSubset');
end