function [  result] = classify_crossValidation(feature, featurei, classid, traj, cv_id, classifier, traj_vote, hier_arch, test_set, option_add)

if ~exist('test_set','var'), test_set = 1;end

if 0 ~= test_set && 1 ~= test_set
    test_set = 1;
end
unused_set = 1 - test_set;

if ~exist('hier_arch','var') || (isstruct(hier_arch) && exist('hier_arch.invalid', 'var')) || (isnumeric(hier_arch) && hier_arch == 0)
    clear hier_arch;
    hier_arch.invalid = 1;
end

if ~exist('classifier','var')
    classifier = 2;
end

if ~exist('option_add', 'var')
    option_add = '';
end

%use trajctery result to vote prediction
if ~exist('traj_vote','var')
    traj_vote = 0;
end
result.b_combine_traj = traj_vote;

%if cv_id is a number, then it is the fold. generate cv_id.
if length(cv_id) == 1
    fold = cv_id;
    traj_id = unique(traj);
    traj_num = length(traj_id);
    traj_class = zeros(traj_num, 1);
    for i = 1:traj_num
        temp_class = classid(traj==traj_id(i));
        traj_class(i) = temp_class(1);
    end
    rand('state', 1);
    traj_indices = crossvalind('Kfold',traj_class, fold);
    
    cv_id = zeros(length(traj), 1);
    for i = 1:fold
        cv_id(ismember(traj, traj_id(traj_indices==i))) = i;
    end
end

cv_id_set = unique(cv_id);
fold = length(cv_id_set);

class_predict = [];
class_result = [];
count_cv_id = [];
scores = [];
test_fish = [];
classify_history = {};

countrecall_i = zeros(fold, 1);    
classprecision_i = zeros(fold, 1);
classrecall_i = zeros(fold, 1);

hier_result = 0;

for i = 1:fold
    tmp_test_id = mod(i+test_set-1, fold)+1;
    tmp_unused_id = mod(i+unused_set-1, fold)+1;
    
    fish_test = (cv_id==cv_id_set(tmp_test_id));
    fish_unused = (cv_id==cv_id_set(tmp_unused_id));
    fish_train = ~fish_test & ~fish_unused;
    
    fprintf('%s Prcessing fold %d/%d\t train: %d\ttest: %d\n', append_timeString(), i, fold, sum(fish_train), sum(fish_test));
 
    hier_arch_i = 0;
    tmp_classify_history = cell(sum(fish_test), 4);
    
    if (exist('hier_arch','var')) && (~isfield(hier_arch, 'invalid'))
        [tmp_predict, tmp_gtclass, tmp_scores, hier_arch_i] = classify_HierSVM(feature(fish_train,:), classid(fish_train), feature(fish_test,:), classid(fish_test), featurei, hier_arch, classifier, option_add);
        if isfield(hier_arch_i, 'Tree_Predict_history')
            tmp_classify_history = hier_arch_i.Tree_Predict_history;
        end
        
        hier_result = inner_addHier(hier_arch_i, hier_result);
    else
%        fish_train = inner_qualityAssess(fish_train, classid, traj, quality);
        [tmp_predict, tmp_scores] = classify_SVM(feature(fish_train,:), classid(fish_train), feature(fish_test,:), classifier, option_add);
        tmp_gtclass = classid(fish_test);
    end
    
    if 1 == result.b_combine_traj
        tmp_predict =  result_trajvote( tmp_predict, traj(fish_test), max(cell2mat(tmp_scores), [], 2) );
    end
    
    test_fish_id = find(fish_test);
    test_number = length(test_fish_id);
    
    count_cv_id(end+1:end+test_number, 1) = cv_id_set(i);
    class_predict(end+1:end+test_number, 1)=tmp_predict;
    class_result(end+1:end+test_number, 1)=tmp_gtclass;
    test_fish(end+1:end+test_number, 1)=test_fish_id;
    scores(end+1:end+test_number, :)=cell2mat(tmp_scores);
    for j = 1:test_number
        classify_history(end+1, :) = tmp_classify_history(j,:);
    end
    
    tmp_result = result_evaluate(tmp_predict, tmp_gtclass);
    tmp_result.result_hier_arch = hier_arch_i;
    cv_result(i) = tmp_result;
    
    countrecall_i(i,1) = cv_result(i).recallCount;    
    classprecision_i(i,1) = cv_result(i).precisionAver;
    classrecall_i(i,1) = cv_result(i).recallAver;
    
    classprecision_all_i(i,:) = cv_result(i).classprecision;
    classrecall_all_i(i,:) = cv_result(i).classrecall;
    
    fprintf('%s AC: %0.4f \t AP: %0.4f \t AR: %0.4f\n', append_timeString(), countrecall_i(i,1), classprecision_i(i,1), classrecall_i(i,1));

%     if cv_one_fold
%         fprintf('only did one fold cross validation\n');
%         countrecall_i(i,1) = countrecall_i(i,1) * fold;    
%         classprecision_i(i,1) = classprecision_i(i,1) * fold;
%         classrecall_i(i,1) = classrecall_i(i,1) * fold;
%         break;
%     end
end

result.count_result = result_evaluate(class_predict, class_result);
result.count_cv_id = count_cv_id;
result.count_class_predict = class_predict;
result.count_class_result = class_result;
result.count_fish_id = test_fish;
result.count_score = scores;
result.count_classify_history = classify_history;
result.count_combine = [traj(test_fish),test_fish,class_predict,class_result,scores];

result.count_result = result_evaluate(result.count_class_predict, result.count_class_result);

result.count_count_recall = result.count_result.recallCount;
result.count_class_precision = result.count_result.precisionAver;
result.count_class_recall = result.count_result.recallAver;

% count_recall = result.totalresult.recallCount;    
% class_precision = result.totalresult.precisionAver;
% class_recall = result.totalresult.recallAver;
count_recall = mean(countrecall_i);
class_precision = mean(classprecision_i);
class_recall = mean(classrecall_i);

classprecision_all_i(end+1,:) = mean(classprecision_all_i);
classprecision_all_i(end+1,:) = result.count_result.classprecision;
classrecall_all_i(end+1,:) = mean(classrecall_all_i);
classrecall_all_i(end+1,:) = result.count_result.classrecall;

result.cv_result = cv_result;
result.cv_count_recall = count_recall;
result.cv_class_precision = class_precision;
result.cv_class_recall = class_recall;
result.cv_class_precision_all = classprecision_all_i;
result.cv_class_recall_all = classrecall_all_i;

result.hier_result = hier_result;


fprintf('%s OverAll: \t AC: %0.4f \t AP: %0.4f \t AR: %0.4f\n\n', append_timeString(), count_recall, class_precision, class_recall);

end

function hier_result = inner_addHier(hier_tree, hier_result)
    if isempty(hier_tree) || ~isfield(hier_tree, 'Tree_predict')
        hier_result = {};
        return;
    end
    
        
    if ~isstruct(hier_result)
        clear hier_result;
        hier_result.Tree_predict = hier_tree.Tree_predict;
        hier_result.Tree_gtclass = hier_tree.Tree_gtclass;
        
        hier_result.Node_predict = hier_tree.Node_predict;
        hier_result.Node_gtclass = hier_tree.Node_classid_test;
        
        for j = 1:length(hier_tree.branch_link)
            hier_result.branch_link{j} = inner_addHier(hier_tree.branch_link{j}, 0);
        end
        
        if isfield(hier_tree, 'Root') && isfield(hier_tree, 'RootRetry_predict')
            hier_result.Root.RootRetry_predict = hier_tree.RootRetry_predict;
            hier_result.Root.RootRetry_gtclass = hier_tree.Tree_gtclass(hier_tree.RootRetry_index);
        end
        
        if isfield(hier_tree, 'GMMResult')
            hier_result.GMMResult = hier_tree.GMMResult;
        end
        
        return;
    end

    samples = length(hier_tree.Tree_predict);
    
    hier_result.Tree_predict(end+1:end+samples) = hier_tree.Tree_predict;
    hier_result.Tree_gtclass(end+1:end+samples) = hier_tree.Tree_gtclass;
    
    hier_result.Node_predict(end+1:end+samples) = hier_tree.Node_predict;
    hier_result.Node_gtclass(end+1:end+samples) = hier_tree.Node_classid_test;

    %recalculate the confus matrix and evaluation result after each
    %inseration.
    tmp_result = result_evaluate(hier_result.Tree_predict, hier_result.Tree_gtclass);
    name_fieds = fieldnames(tmp_result);
    for i=1:length(name_fieds)
        hier_result.(char(name_fieds(i))) =  tmp_result.(char(name_fieds(i)));
        %assignin('caller', hier_result.(char(name_fieds(i))), tmp_result.(char(name_fieds(i))));
    end
    
    for i = 1:length(hier_tree.branch_link)
        hier_result.branch_link{i} = inner_addHier(hier_tree.branch_link{i}, hier_result.branch_link{i});
    end
    
    if isfield(hier_tree, 'Root') && isfield(hier_tree, 'RootRetry_predict')
        rootSamples = length(hier_tree.RootRetry_predict);
        hier_result.Root.RootRetry_predict(end+1:end+rootSamples) = hier_tree.RootRetry_predict;
        hier_result.Root.RootRetry_gtclass(end+1:end+rootSamples) = hier_tree.Tree_gtclass(hier_tree.RootRetry_index);
        
        tmp_result = result_evaluate(hier_result.Root.RootRetry_predict, hier_result.Root.RootRetry_gtclass);
        name_fieds = fieldnames(tmp_result);
        for i=1:length(name_fieds)
            hier_result.Root.(char(name_fieds(i))) =  tmp_result.(char(name_fieds(i)));
        end
    end
    
    if isfield(hier_tree, 'GMMResult')
        hier_result.GMMResultComplete = [];
        GMMcomponent = length(hier_tree.GMMResult);
        for j = 1:GMMcomponent
            [samples, speciesNum] = size(hier_tree.GMMResult{j});
            hier_result.GMMResult{j}(end+1:end+samples,:) = hier_tree.GMMResult{j};
            
            hier_result.GMMResultComplete{j}(:,1) = hier_result.Tree_predict;
            hier_result.GMMResultComplete{j}(:,2) = hier_result.Tree_gtclass;
            for s = 1:size(hier_result.GMMResult{j}, 1)
                [hier_result.GMMResultComplete{j}(s,3), hier_result.GMMResultComplete{j}(s,4)] = max(hier_result.GMMResult{j}(s,:));
            end
            hier_result.GMMResultComplete{j}(:,5:speciesNum+4) = hier_result.GMMResult{j};
        end
    end
end

function train_id = inner_qualityAssess(fish_train, classid, traj, quality)
fish_train = find(fish_train);
classid = classid(fish_train);
%traj = traj(fish_train);
%quality = quality(fish_train);

indic1 = find(classid == 1);
indic2 = find(classid ~= 1);

traj_id1_uni = unique(traj(fish_train(indic1)));
valid_indic = [];
for i = 1:length(traj_id1_uni)
tmp_indic = find(traj==traj_id1_uni(i));
[mv,mi]=max(quality(tmp_indic));
valid_indic(end+1,1)=tmp_indic(mi);
end

rand_indic = randperm(length(valid_indic));
train_id = [fish_train(indic2); valid_indic(rand_indic(1:100))];
end