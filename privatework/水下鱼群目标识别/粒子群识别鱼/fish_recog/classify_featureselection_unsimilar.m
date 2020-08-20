function [featureSubset, scoreResult] = classify_featureselection_unsimilar(features, feature_id, classid, traj, maxfeature, standard, fold, classifier, filesavename, traj_vote, hier_tree, cv_one_fold, unsimilarSize, testSimMethod, indiv_performance)

if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

if ~exist('unsimilarSize','var')
    unsimilarSize = 3 * maxfeature;
end

if ~exist('traj_vote','var')
    traj_vote = 0;
end

if ~exist('testSimMethod','var')
    testSimMethod = 1;
    % 1 test similarity by feature
    % 2 test similarity by result
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

feature_left = ones(featuresnum, 1);
if exist([filesavename,'.mat'],'file')
    tmp_data = load(filesavename);
    ctime = clock();
    old_filesavename = sprintf('%s_%d_%d_%d_%d_%d_old.mat', filesavename, ctime(1), ctime(2), ctime(3), ctime(4), ctime(5));
    movefile([filesavename,'.mat'], old_filesavename);
    
    scoreResult = tmp_data.scoreResult;
    featureSubset = tmp_data.featureSubset;
    loopstart = length(featureSubset) + 1;
    feature_left = tmp_data.feature_left;
else
    if testSimMethod == 1
        similarity = corrcoef(features);
        similarity(isnan(similarity))=0;
        
        tmp_threshold = 0.5;
        for i = 1:featuresnum
            if ~feature_left(i)
                continue;
            end
            
            candidate = find(abs(similarity(i,:)) > tmp_threshold & feature_left(i));
            [mv,mi]=max(indiv_performance(candidate));
            feature_left(candidate) = 0;     
            feature_left(candidate(mi)) = 1;
        end
        
    elseif testSimMethod == 2
        similarity = append_CalculateSim_byResult(feature, classid);
    end
    
    save(filesavename, 'featureSubset', 'scoreResult', 'feature_left');
end

for loop1 = loopstart:maxfeature
    fprintf('\nBegin %d/%d\n', loop1, maxfeature);

    tmpFeatureSubset = zeros(featuresnum,length(featureSubset)+1);
    tmpval = zeros(featuresnum,3);
    
    for loop2_i = 1:featuresnum
        loop2 = feature_hist(loop2_i);
        if(find(featureSubset==loop2))
            continue;
        end
        
        if ~feature_left(loop2_i)
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
            
        fprintf('select: %d/%d test %d/%d id:%d count_recall:%f class_precision:%f class_recall:%f\n', loop1, maxfeature, loop2_i, featuresnum, loop2, tmpval(loop2_i, 1), tmpval(loop2_i, 2), tmpval(loop2_i, 3));
    end
    [currentbest, bestIndex] = max(tmpval(:,standard));
    
    featureSubset = tmpFeatureSubset(bestIndex,:);
    scoreResult(loop1,:) = tmpval(bestIndex, :);

    fprintf('Choosed %d\n Score: %f count_recall:%f class_precision:%f class_recall:%f\n', bestIndex, currentbest, tmpval(bestIndex, 1), tmpval(bestIndex, 2), tmpval(bestIndex, 3));
    fprintf('Max value: count_recall:%f class_precision:%f class_recall:%f\n', max(tmpval(:,1)), max(tmpval(:,2)), max(tmpval(:,3)));

    save(filesavename, 'featureSubset', 'scoreResult', 'feature_left');
end

end