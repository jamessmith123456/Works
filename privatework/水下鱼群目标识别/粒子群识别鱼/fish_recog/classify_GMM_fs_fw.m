function [featureSubset, scoreResult] = classify_GMM_fs_fw(features, featurei, class_id, valid_class, maxfeature, cv_id, filesavename)

if ~exist('filesavename','var')
    filesavename = 'GMM_featureselection.mat';
end

%use validatation set as test set
test_set = 0;

if ~exist('cv_id','var')
    cv_id = 5;
end

feature_hist = unique(featurei);
featuresnum = length(feature_hist);
if ~exist('maxfeature','var')
    maxfeature = featuresnum;
end
maxfeature = min(maxfeature, featuresnum);

featureSubset = [];
loopstart = 1;
scoreResult = zeros(maxfeature, 1);

if exist([filesavename,'.mat'],'file') || exist(filesavename,'file')
    tmp_data = load(filesavename);
    ctime = clock();
    old_filesavename = sprintf('%s_%d_%d_%d_%d_%d_old.mat', filesavename, ctime(1), ctime(2), ctime(3), ctime(4), ctime(5));
    
    if exist([filesavename,'.mat'],'file')
        copyfile([filesavename,'.mat'], old_filesavename);
    elseif exist(filesavename,'file')
        copyfile(filesavename, old_filesavename);
    end
    
    scoreResult = tmp_data.scoreResult;
    featureSubset = tmp_data.featureSubset;
    loopstart = length(featureSubset) + 1;
end

for loop1 = loopstart:maxfeature
    fprintf('\n%s Begin %d/%d\n', append_timeString(), loop1, maxfeature);

    subsetsize = length(featureSubset)+1;
    tmpFeatureSubset = zeros(featuresnum, subsetsize);

    tmpval = -inf * ones(featuresnum,1);
    
    for loop2_i = 1:featuresnum
        loop2 = feature_hist(loop2_i);
        if(find(featureSubset==loop2))
            continue;
        end
        
        for i = 1:length(featureSubset)
            tmpFeatureSubset(loop2_i,i)=featureSubset(i);
        end
        tmpFeatureSubset(loop2_i,subsetsize) = loop2;
        
        subset_index = ismember(featurei, tmpFeatureSubset(loop2_i,:));
        subset_feature = features(:, subset_index);
%        subset_id = featurei(subset_index);
        
        tmp_result = classify_GMM_crossValidation(subset_feature, class_id, valid_class, cv_id, test_set);
        tmpval(loop2_i, 1) = tmp_result.averDistance;
            
        fprintf('%s loop %d/%d\ttest %d/%d id:%d\tdiffer: %f\n', append_timeString(), loop1, maxfeature, loop2_i, featuresnum, loop2, tmpval(loop2_i, 1));
    end
    [currentbest, bestIndex] = max(tmpval);
    
    featureSubset = tmpFeatureSubset(bestIndex,:);
    scoreResult(loop1,:) = tmpval(bestIndex, :);

    fprintf('Choosed %d\n Score: %f\n', bestIndex, currentbest);
    save(filesavename, 'featureSubset', 'scoreResult');
end

end