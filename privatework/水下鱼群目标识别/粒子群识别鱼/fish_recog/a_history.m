[mv,mi]=min(score_AIC, [], 1);
[mv,mi]=min(score_AIC, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(score_BIC, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
score_model = reshape(score_model, 15,50);
model_BIC = score_model(5,1);
model_AIC = score_model(34,13);
model_AIC = score_model(13, 34);
score_liklyhood_test_AIC = classify_GMM_predict(features_minor(result_test_set(1,1).minor_predict==i,:),model_AIC);
score_liklyhood_test_AIC = classify_GMM_predict(features_minor(result_test_set(1,1).minor_predict==1,:),model_AIC);
score_liklyhood_test_AIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1)),model_AIC);
score_liklyhood_test_AIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1),:),model_AIC);
score_liklyhood_test_AIC = classify_GMM_predict(features_minor(result_test_set(1,1).minor_predict==1,:),model_AIC);
score_liklyhood_self_AIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1),:),model_AIC);
[ tpr,fpr,thresholds ] = append_CalcROCFromSeperateArray( score_liklyhood_self_AIC, score_liklyhood_test_AIC )
score_liklyhood_test_BIC = classify_GMM_predict(features_minor(result_test_set(1,1).minor_predict==1,:),model_BIC);
score_liklyhood_self_BIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1),:),model_BIC);
[ tpr,fpr,thresholds ] = append_CalcROCFromSeperateArray( score_liklyhood_self_BIC, score_liklyhood_test_BIC );
[ tpr,fpr,thresholds ] = append_CalcROCFromSeperateArray( score_liklyhood_self_AIC, score_liklyhood_test_AIC )
score_liklyhood_self_AIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1) & classid_major(result_test_set(1,1).test_id)==1,:),model_AIC);
score_liklyhood_self_AIC = classify_GMM_predict(features_major(result_test_set(1,1).test_id(result_test_set(1,1).test_predict==1 & classid_major(result_test_set(1,1).test_id)==1),:),model_AIC);
[ tpr,fpr,thresholds ] = append_CalcROCFromSeperateArray( score_liklyhood_self_AIC, score_liklyhood_test_AIC )
help trapz
trapz(fpr, tpr)
trapz(tpr, fpr)
[auc ] = append_CalcROCFromSeperateArray( score_liklyhood_self_AIC, score_liklyhood_test_AIC )
[result] = rejection_Unsupervised_Train(features_major(result_test_set(1,1).train_id, :), classid_major(result_test_set(1,1).train_id, :));
[result] = rejection_Unsupervised_Train(features_major(result_test_set(1,1).train_id, :), classid_major(result_test_set(1,1).train_id, :), featurei, featureSubset);
load('rejection_data_7k2_BGOT.mat', 'featureSubset', 'featurei')
[result] = rejection_Unsupervised_Train(features_major(result_test_set(1,1).train_id, :), classid_major(result_test_set(1,1).train_id, :), featurei, featureSubset);
rejection_script_Unsupervised
sum(classid_train == index_species)
rejection_script_Unsupervised
%-- 2013-03-04 11:41 PM --%
load('rejection_data_7k2_BGOT_20130304.mat')
rejection_script_Unsupervised
load('1.fish7k2_feature_20120409.mat', 'feature')
load('1.fish7k2_feature_20130205_normalized.mat', 'features')
[features]= interface_normalizeFeatureSet(feature);
[features, features_minor2]= interface_normalizeFeatureSet(feature, features_minor);
rejection_script_Unsupervised
%-- 2013-03-05 02:27 PM --%
load('matlab.mat', 'result_AIC', 'result_BIC')
for i = 1:4
auc_AIC{i}=[auc_AIC{i},result_BIC(:).auc(i)];
end
for i = 1:5
for j = 1:4
auc(i,j)=result_BIC(i).auc(j);
end
end
for i = 1:5
for j = 1:4
auc_AIC(i,j)=result_AIC(i).auc(j);
end
end
mean(auc_AIC)
mean(auc_BIC)
std(auc_AIC)
std(auc_BIC)
for i = 1:5
for j = 1:4
com_AIC(i,j)=result_AIC(i).model.output_XIC_com(j);
fn_AIC(i,j)=result_AIC(i).model.output_XIC_fn(j);
com_BIC(i,j)=result_BIC(i).model.output_XIC_com(j);
fn_BIC(i,j)=result_BIC(i).model.output_XIC_fn(j);
end
end
mean(com_AIC)
mean(com_BIC)
load('matlab.mat', 'result_AIC', 'result_BIC')
for i = 2:5
for j = 1:4
result_AIC(1).model.output_XIC_array{j} = result_AIC(1).model.output_XIC_array{j} + result_AIC(i).model.output_XIC_array{j};
result_BIC(1).model.output_XIC_array{j} = result_BIC(1).model.output_XIC_array{j} + result_BIC(i).model.output_XIC_array{j};
end
end
[mv,mi]=min(result_AIC(1).model.output_XIC_array{1}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_AIC(1).model.output_XIC_array{2}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_AIC(1).model.output_XIC_array{3}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_AIC(1).model.output_XIC_array{4}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_AIC(1).model.output_XIC_array{3}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_AIC(1).model.output_XIC_array{4}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{1}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{2}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{3}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{2}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{1}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
[mv,mi]=min(result_BIC(1).model.output_XIC_array{4}, [], 2);
[minv, min_com] = min(mv);
min_fn = mi(min_com);
result_AIC(:).model.output_XIC_array{1}
{result_AIC(:).model.output_XIC_array{1}}
{result_AIC(:).model.output_XIC_array(1)}
{result_AIC(:).model.output_XIC_array}
load('rejection_data_7k2_BGOT_20130304.mat')
rejection_script_Unsupervised
[result(:).auc]
mean([result(:).auc])
std([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
std([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
std([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
std([result(:).auc])
rejection_script_Unsupervised
result(:).score_liklyhood_valid
{result(:).score_liklyhood_valid}
{result(:).score_liklyhood_valid{1}}
{result(:).score_liklyhood_valid}
s4_valid = ans;
s4_unknown = {result(:).score_liklyhood_unknown};
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
s3_valid = {s3_result(:).score_liklyhood_valid};
s3_unknown = {s3_result(:).score_liklyhood_unknown};
rejection_script_Unsupervised
mean([result(:).auc])
rejection_script_Unsupervised
mean([result(:).auc])
s2_valid = {s2_result(:).score_liklyhood_valid};
s2_unknown = {s2_result(:).score_liklyhood_unknown};
rejection_script_Unsupervised
mean([result(:).auc])
s1_valid = {s1_result(:).score_liklyhood_valid};
s1_unknown = {s1_result(:).score_liklyhood_unknown};
s1_unknown{1}
s1_unknown{1}{1}
s1_unknown{:}{1}
[s1_unknown{:}{1}]
s1_unknown_all = [];
s1_valid_all = [];
for i = 1:5
s1_unknown_all = [s1_unknown_all; s1_unknown{i}{1}];
s1_valid_all = [s1_valid_all;  s1_valid{i}{1}];
end
s2_unknown_all = [];
s2_valid_all = [];
for i = 1:5
s2_unknown_all = [s2_unknown_all; s2_unknown{i}{1}];
s2_valid_all = [s2_valid_all;  s2_valid{i}{1}];
end
s3_unknown_all = [];
s3_valid_all = [];
for i = 1:5
s3_unknown_all = [s3_unknown_all; s3_unknown{i}{1}];
s3_valid_all = [s3_valid_all;  s3_valid{i}{1}];
end
s4_unknown_all = [];
s4_valid_all = [];
for i = 1:5
s4_unknown_all = [s4_unknown_all; s4_unknown{i}{1}];
s4_valid_all = [s4_valid_all;  s4_valid{i}{1}];
end
append_CalcROCFromSeperateArray( s1_valid_all, s1_unknown_all );
append_CalcROCFromSeperateArray( s2_valid_all, s2_unknown_all );
append_CalcROCFromSeperateArray( s3_valid_all, s3_unknown_all );
append_CalcROCFromSeperateArray( s4_valid_all, s4_unknown_all );
%-- 2013-03-05 04:25 PM --%
load('rejection_data_7k2_BGOT_20130304.mat', 'features_minor')
load('rejection_data_7k2_BGOT_20130304.mat')
load('0.fish7k2_attribute_minor.mat', 'traj_id')
traj = unique(traj_id);
traj_num = length(traj);
traj_indices = crossvalind('Kfold',traj, 2);
traj_train = find(ismember(traj_id, traj(traj_indics==1)));
traj_train = find(ismember(traj_id, traj(traj_indices==1)));
traj_test = find(ismember(traj_id, traj(traj_indices==2)));
unique([traj_train,traj_test]);
unique([traj_train;traj_test]);
cvid_minor = ones(305,1);
cvid_minor(traj_test)=2;
tabulate(cvid_minor)
load('rejection_data_7k2_BGOT_20130304_unsupervised.mat')
valid_set = ones(6874,1);
valid_set(result_test_set(1,1).train_id)=0;
valid_set(result_test_set(1,1).test_id)=0;
sum(valid_set)
result_test_set(1,1).validation_set = find(valid_set==1);
valid_set = ones(6874,1);
valid_set(result_test_set(2).train_id)=0;
valid_set(result_test_set(2).test_id)=0;
result_test_set(2).validation_set = find(valid_set==1);
valid_set = ones(6874,1);
valid_set(result_test_set(3).train_id)=0;
valid_set(result_test_set(3).test_id)=0;
result_test_set(3).validation_set = find(valid_set==1);
valid_set = ones(6874,1);
valid_set(result_test_set(4).train_id)=0;
valid_set(result_test_set(4).test_id)=0;
result_test_set(4).validation_set = find(valid_set==1);
valid_set = ones(6874,1);
valid_set(result_test_set(5).train_id)=0;
valid_set(result_test_set(5).test_id)=0;
result_test_set(5).validation_set = find(valid_set==1);
rejection_script_Supervised_GMM
load('rejection_data_7k2_BGOT_20130304_unsupervised.mat', 'result_validation_set')
result_test_set(1,1).validation_predict = result_validation_set(1,1).test_predict;
result_test_set(2).validation_predict = result_validation_set(2).test_predict;
result_test_set(3).validation_predict = result_validation_set(3).test_predict;
result_test_set(4).validation_predict = result_validation_set(4).test_predict;
result_test_set(5).validation_predict = result_validation_set(5).test_predict;
rejection_script_Supervised_GMM
feature_unknown_train(tmp_index_unkown,:)
model.output_model(index_componentID, index_featureNumID, index_speciesID)
rejection_script_Supervised_GMM
sum(tmp_index_unkown)
sum(tmp_index_valid)
rejection_script_Supervised_GMM
mean(result.auc)
std(result.auc)
for i = 1:4
score_unknown = [];
score_valid = [];
for j = 1:5
score_unknown = [score_unknown; result_sGMM_c5_f10.score_liklyhood_unknown{j, i}];
score_valid = [score_valid; result_sGMM_c5_f10.score_liklyhood_valid{j, i}];
append_CalcROCFromSeperateArray( score_valid, score_unknown, 1 );
pause();
end
end
for i = 1:4
score_unknown = [];
score_valid = [];
for j = 1:5
score_unknown = [score_unknown; result_sGMM_c5_f10.score_liklyhood_unknown{j, i}];
score_valid = [score_valid; result_sGMM_c5_f10.score_liklyhood_valid{j, i}];
append_CalcROCFromSeperateArray( score_valid, score_unknown, 1 );
end
pause();
end
for i = 1:4
score_unknown = [];
score_valid = [];
for j = 1:5
score_unknown = [score_unknown; result_sGMM_c5_f10.score_liklyhood_unknown{j, i}];
score_valid = [score_valid; result_sGMM_c5_f10.score_liklyhood_valid{j, i}];
append_CalcROCFromSeperateArray( score_valid, score_unknown, 1 );
end
pause();
end
for i = 1:4
score_unknown = [];
score_valid = [];
for j = 1:5
score_unknown = [score_unknown; result_sGMM_c5_f10.score_liklyhood_unknown{j, i}];
score_valid = [score_valid; result_sGMM_c5_f10.score_liklyhood_valid{j, i}];
end
append_CalcROCFromSeperateArray( score_valid, score_unknown, 1 );
pause();
end
load('rejection_data_7k2_BGOT_20130304_unsupervised.mat')
clear;
load('rejection_data_7k2_BGOT_20130305_supervised.mat')
rejection_script_Supervised_SVM
rejection_Supervised_SVM_script
clear tmp_*;
rejection_script_Supervised_GMM
rejection_Supervised_GMM_script
clear;
%-- 2013-03-07 02:24 PM --%
load('2.fish90k_feature_23species_130226_normalize.mat')
help save
hemp combnk
help combnk
combnk(3,5)
combnk(5,3)
combnk([1:5],3)
c = combnk([1:23],11);
result = append_testNodeSplit( features, 1:2626, class_id, traj_id, 5, 2, 0, [1:23], 1, 1000)
result = append_testNodeSplit( features, 1:2626, class_id, traj_id, 5, 2, 0, [1:23], 1, 100000)
load('2.fish90k_feature_23species_130226_normalize.mat', 'cv_id_fold5')
train_test_id = cv_id_fold5 == 5;
[ result ] = append_testNodeSplit( features, class_id, train_test_id, 2, [1:23], 1, 100000);
sum(tmp_class_indic)
sum(tmp_classid_train)
sum(tmp_classid_test)
sum(~tmp_classid_test)
sum(~tmp_classid_train)
c = combnk([1:23],11);
2^23
[ result ] = append_testNodeSplit( features(:,1:500), class_id, train_test_id, 2, [1:23], 1, 100000);
make
mex -setup
[ result ] = append_testNodeSplit( features(:,1:500), class_id, train_test_id, 2, [1:23], 1, 100000);
[ result ] = append_testNodeSplit( features(:,1:500), class_id, train_test_id, 2, [1:23], 1, 100000, 'tmpsave.mat');
clear;
2^23
script_NodeSplit_N1( 1, 1000 )
pwd
fullfile(pwd, '/123.mat')
deploytool
help bwlabel
[ result ] = script_NodeSplit_N1( 1 )
%-- 2013-03-12 03:47 PM --%
load('1.fish7k2_feature_20130205_normalized.mat', 'features')
featurei = 1:2626;
load('1.fish7k2_feature_20130205_normalized.mat', 'traj_id', 'class_id')
class_id(class_id>2)=2;
[featureSubset_1, scoreResult_1] = classify_featureselection_fw(features, featurei, class_id, traj_id, 500, 3, 5, 2, 'feature_1vsRest');
clear;
script_featureSelection_1vsRest
script_featureSelection_1vsRest(1)
license ('test','Compiler')
ver
license('test','Distrib_Computing_Toolbox')
script_featureSelection_1vsRest(1)
%-- 2013-03-12 09:08 PM --%
script_featureSelection_1vsRest(2)
load('rejection_data_7k2_BGOT_20130305_supervised.mat')
rejection_Supervised_SVM_script
clear;
%-- 2013-03-13 11:43 PM --%
%-- 2013-03-14 12:30 AM --%
load('2.fish90k_feature_23species_130226_normalize.mat')
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0)
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 3, 1)
help svmtrain
%-- 2013-03-14 01:51 AM --%
load('9.result_20130313_species1234.mat', 'result_sSVM')
load('1.data_7k2_BGOT_20130305_supervised.mat')
rejection_Supervised_SVM_script
%-- 2013-03-14 01:56 AM --%
script_featureSelection_1vsRest( 3 )
%-- 2013-03-20 02:52 PM --%
script_featureSelection_1vsRest(4)
load('1.fish7k2_feature_20130205_normalized.mat')
classify_featureselection_pool(feature, 1:2626, class_id, traj_id, 100, 3, 5, 2, 'save_individual');
classify_featureselection_pool(features, 1:2626, class_id, traj_id, 100, 3, 5, 2, 'save_individual');
save('save_individual', 'featurePool');
load('save_individual.mat')
save('save_individual', 'featurePool');
save('save_individual', 'featurePool2', '-append');
classify_featureselection_pool(features, 1:2626, class_id, traj_id, 100, 3, 5, 2, 'save_individual');
load('save_individual.mat', 'indiv_performance')
%-- 2013-03-26 05:43 PM --%
%-- 2013-03-26 06:16 PM --%
load('2.fish90k_feature_23species_130226_normalize.mat', 'class_id', 'traj_id', 'featurei', 'features')
[  result] = classify_crossValidation(features, 1:2626, class_id, traj_id, 5, 2)
%-- 2013-03-28 02:52 PM --%
load('matlab.mat', 'result')
[ pairs_top2 ] = append_findSimilarPair_top2(result.count_result.confus);
[ pairs_self ] = append_findSimilarPair_self(result.count_result.confus);
[ pairs_coSim ] = append_findSimilarPair_coSim(result.count_result.confus);
[ pairs_top2 ] = append_findSimilarPair_top2(result.count_result.confus);
[ pairs_self ] = append_findSimilarPair_self(result.count_result.confus);
[ pairs_coSim ] = append_findSimilarPair_coSim(result.count_result.confus);
[ pairs_self ] = append_findSimilarPair_self(result.count_result.confus);
[ pairs_top2 ] = append_findSimilarPair_top2(result.count_result.confus);
[ pairs_self ] = append_findSimilarPair_self(result.count_result.confus);
[ pairs_top2 ] = append_findSimilarPair_top2(result.count_result.confus);
load('1.fish7k2_feature_20130205_normalized.mat')
[  fish_7k2_result] = classify_crossValidation(features, 1:2626, class_id, traj_id, 5, 2, 0, 0, 0, 1)
[  fish_7k2_result] = classify_crossValidation(features, 1:2626, class_id, traj_id, 5, 2, 0, 0, 0, 0)
[ fish_7k2_pairs_coSim ] = append_findSimilarPair_coSim(fish_7k2_result.count_result.confus);
[ fish_7k2_pairs_self ] = append_findSimilarPair_self(fish_7k2_result.count_result.confus);
[ fish_7k2_pairs_top2 ] = append_findSimilarPair_top2(fish_7k2_result.count_result.confus);
load('2.fish90k_feature_23species_130226_normalize.mat')
[  fish_7k2_result] = classify_crossValidation(features, 1:2626, class_id, traj_id, 5, 2, 0, 0, 0, 0)
[ fish_27k_pairs_coSim ] = append_findSimilarPair_coSim(fish_27k_result.count_result.confus);
[ fish_27k_pairs_self ] = append_findSimilarPair_self(fish_27k_result.count_result.confus);
[ fish_27k_pairs_top2 ] = append_findSimilarPair_top2(fish_27k_result.count_result.confus);
load('0.fish7k2_attribute.mat', 'traj_id', 'class_id', 'cv_id_fold5')
load('1.fish7k2_feature_20130205_normalized.mat', 'traj_id', 'class_id', 'features')
128
2^14
2^7
nchoosek(15,7)
[ result_group0 ] = append_testNodeSplit( features, 1:2626, class_id, traj_id, cv_id_fold5, 1:15, 1, 16384, 'result_group0');
nchoosek(15,7)
6435/60
6435/60/24
16384/4
load('4.fish27k_testSplit_flatSVM.mat', 'fish_27k_result')
[ fish_27k_pairs_top2 ] = append_findSimilarPair_top2(fish_27k_result.count_result.confus);
find(diag(confus)~=0)
4096*2
4096*3
load('0.fish7k2_attribute.mat', 'cv_id_fold5')
load('1.fish7k2_feature_20130205_normalized.mat')
4096*4
[ result_group0_4 ] = append_testNodeSplit( features, 1:2626, class_id, traj_id, cv_id_fold5, 1:15, 12289, 16384, 'result_group0_4');
[ result_group0_2 ] = append_testNodeSplit( features, 1:2626, class_id, traj_id, cv_id_fold5, 1:15, 4097, 8192, 'result_group0_2');
[ result_group0_2 ] = append_testNodeSplit( features, 1:2626, class_id, traj_id, cv_id_fold5, 1:15, 4097, 8192, 'result_group0_2.mat');
%-- 2013-03-29 06:24 PM --%
load('0.fish7k2_attribute.mat', 'cv_id_fold5')
load('1.fish7k2_feature_20130205_normalized.mat')
[ result_group0_3 ] = append_testNodeSplit( features, 1:2626, class_id, traj_id, cv_id_fold5, 1:15, 8193, 12288, 'result_group0_3.mat');
%-- 2013-03-30 04:15 PM --%
load('nodeSplit_0006_128.mat', 'result')
mergedResult=append_mergeSplitResult(128);
max(mergedResult_byGraph(:,3))
max(mergedResult_byTop2(:,3))
max(mergedResult_top2(:,3))
[mv,mi]=max(mergedResult_top2(:,3))
[mv,mi]=max(mergedResult_byGraph(:,3))
de2bi(287,12)
mergedResult_byGraph(287,:)
branch_class{1,1}=[1,2,3,7,8,9,14,15,17,18,21,22];
branch_class{2,1}=[4,5,6,10,11,12,13,16,19,20,23];
load('2.fish90k_feature_23species_130226_normalize.mat')
indic = ismember(class_id, branch_class{1});
class_id = class_id(indic);
featurei = 1:2626;
features = features(indic,:);
traj_id = traj_id(indic);
cv_id_fold5 = cv_id_fold5(indic);
tabulate(class_id)
for i = 1:12
class_id(class_id==branch_class{1}(i))=i;
end
tabulate(class_id)
[ result ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, 1:12, 1, 2048, 'node2_byGraph');
%-- 2013-03-31 08:48 PM --%
load('result_group_27k.mat')
load('4.fish7k2_testSplit_node1.mat', 'mergedResult_byGraph')
clear;
load('4.fish7k2_testSplit_node1.mat')
clear;
[ result ] = script_NodeSplit_N2_reGroup( 5 )
[ result ] = script_NodeSplit_N2_reGroup( 4 )
load('2.fish90k_feature_23species_130226_normalize.mat')
load('4.fish27k_testSplit_node1.mat', 'branch_class_byGraph')
class_id(ismember(class_id,branch_class_byGraph{1}))=1;
class_id(ismember(class_id,branch_class_byGraph{2}))=2;
tabulate(class_id);
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, 5, 2, 'fs_node1_byGraph', 0, 0, 0, cv_id_fold5)
%-- 2013-04-04 05:04 PM --%
branch_class{1,1}=[1,2,3,7,8,9,14,15,17,18,21,22];
branch_class{2,1}=[4,5,6,10,11,12,13,16,19,20,23];
load('2.fish90k_feature_23species_130226_normalize.mat')
load('4.fish27k_testSplit_node1.mat', 'branch_class_byGraph')
class_id(ismember(class_id,branch_class_byGraph{1}))=1;
class_id(ismember(class_id,branch_class_byGraph{2}))=2;
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, 5, 2, 'fs_node1_byGraph', 0, 0, 0, cv_id_fold5)
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, 5, 2, '3.fish27k_fs_hierTree1301_noe1', 0, 0, 0, cv_id_fold5)
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 1000, 3, cv_id_fold5, 2, '3.fish27k_fs_hierTree1301_noe1', 0, 0)
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 1000, 3, cv_id_fold5, 2, '3.fish27k_fs_hierTree1301_noe1.mat', 0, 0)
%-- 2013-04-16 10:41 PM --%
branch_class{1,1}=[1,2,3,7,8,9,14,15,17,18,21,22];
branch_class{2,1}=[4,5,6,10,11,12,13,16,19,20,23];
load('2.fish90k_feature_23species_130226_selfNormalize95.mat')
load('4.fish27k_testSplit_node1.mat', 'branch_class_byGraph')
class_id(ismember(class_id,branch_class_byGraph{1}))=1;
class_id(ismember(class_id,branch_class_byGraph{2}))=2;
[featureSubset, scoreResult] = classify_featureselection_fw(features, featurei, class_id, traj_id, 1000, 3, cv_id_fold5, 2, '3.fish27k_fs_hierTree1301_node1_69types.mat', 0, 0)
max(scoreResult)
%-- 2013-04-30 11:24 AM --%
load('4.fish27k_hierTree_1301_20130430.mat')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'featurei', 'cv_id_fold5', 'features')
[result_1301a_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301a, 0)
[result_1301b_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301b, 0)
[result_1301c_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301c, 0)
[result_1301d_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301d, 0)
[result_1302a_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1302a, 0)
[result_1301_69a_0] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301_69a, 0)
[result_1301a_1] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301a, 1)
[result_1301b_1] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301b, 1)
[result_1301c_1] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301c, 1)
[result_1301d_1] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301d, 1)
[result_1302a_1] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1302a, 1)
[result_1301_69a_1] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301_69a, 1)
[result_1301_69a_0] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301_69a, 0)
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a', 'hier_tree_1301_69b')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301a', 'hier_tree_1301d', 'hier_tree_1301b', 'hier_tree_1301c', 'hier_tree_1302a', 'hier_tree_1301_69a', 'hier_tree_1301_69b')
[result_1301_69c_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 0, hier_tree_1301_69c, 0)
[result_1301_69c_0] = classify_crossValidation(features, 1:2626, class_id, traj_id, cv_id_fold5, 2, 1, hier_tree_1301_69c, 0)
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
[result_1301_69a_0] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 1, hier_tree_1301_69a, 0)
%-- 2013-04-30 10:47 PM --%
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'featurei', 'cv_id_fold5', 'features')
help crossvalind
tmp = crossvalind('Kfold',1:100, 2);
for i = 1:5
tmp_indic = cv_id_fold5 == i;
downsample_fold2(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 2);
downsample_fold3(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 3);
end
for i = 1:5
tmp_indic = cv_id_fold5 == i&class_id == 1;
downsample_fold2(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 2);
downsample_fold3(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 3);
end
downsample_fold2 = zeros(27370,1);
downsample_fold3 = zeros(27370,1);
for i = 1:5
tmp_indic = cv_id_fold5 == i&class_id == 1;
downsample_fold2(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 2);
downsample_fold3(tmp_indic) = crossvalind('Kfold',1:sum(tmp_indic), 3);
end
tabulate(downsample_fold2)
tabulate(downsample_fold3)
tabulate(class_id)
tabulate(class_id(downsample_fold2<2))
indic_fold2 = downsample_fold2<2;
indic_fold3 = downsample_fold3<3;
indic_fold4 = downsample_fold3<2;
[result_1301_69a_1_2] = classify_crossValidation(features(indic_1_2,:), featurei(indic_1_2), class_id(indic_1_2), traj_id(indic_1_2), cv_id_fold5(indic_1_2), 2, 0, hier_tree_1301_69a, 0)
[result_1301_69a_1_3] = classify_crossValidation(features(indic_1_3,:), featurei(indic_1_3), class_id(indic_1_3), traj_id(indic_1_3), cv_id_fold5(indic_1_3), 2, 0, hier_tree_1301_69a, 0)
[result_1301_69a_2_3] = classify_crossValidation(features(indic_2_3,:), featurei(indic_2_3), class_id(indic_2_3), traj_id(indic_2_3), cv_id_fold5(indic_2_3), 2, 0, hier_tree_1301_69a, 0)
[result_1301_69a_1_2] = classify_crossValidation(features(indic_1_2,:), featurei, class_id(indic_1_2), traj_id(indic_1_2), cv_id_fold5(indic_1_2), 2, 0, hier_tree_1301_69a, 0)
[result_1301_69a_1_3] = classify_crossValidation(features(indic_1_3,:), featurei, class_id(indic_1_3), traj_id(indic_1_3), cv_id_fold5(indic_1_3), 2, 0, hier_tree_1301_69a, 0)
[result_1301_69a_2_3] = classify_crossValidation(features(indic_2_3,:), featurei, class_id(indic_2_3), traj_id(indic_2_3), cv_id_fold5(indic_2_3), 2, 0, hier_tree_1301_69a, 0)
%-- 2013-05-02 08:38 AM --%
load('data_featureSelection_Species_0001.mat')
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv(i,1),mi(i,1)=max(scoreResult(:,3));
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv(i,1),mi(i,1)]=max(scoreResult(:,3));
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv(i,1),mi(i,1)]=max(tmp_data.scoreResult(:,3));
featureSub{i,1}=tmp_data.featureSubset(1:mi(i,1));
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=tmp_data.featureSubset(1:mi(i,1));
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=tmp_data.featureSubset(1:mi(i,1));
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=tmp_data.featureSubset(1:mi(i,1));
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=tmp_data.featureSubset(1:mi(i,1));
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=tmp_data.featureSubset(1:featureSub{i,2});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[featureSub{i,1},featureSub{i,2}]=max(tmp_data.scoreResult(:,3));
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=tmp_data.featureSubset(1:featureSub{i,2});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
load('data_all_summary.mat')
featureSub(:,7:12)=featureSub_69;
load('data_all_summary.mat', 'featureSub')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'featurei')
for i = 1:15
featureSub{4,i}=find(ismember(featurei, featureSub{4,i}));
featureSub{2,i} = length(featureSub{4,i});
end
load('data_all_summary.mat')
featureSub{4,i}=find(ismember(featurei, featureSub{4,i}));
find(ismember(featurei, featureSub{4,i}));
featureSub{4,i}
for i = 1:15
featureSub{i,4}=find(ismember(featurei, featureSub{i,4}));
featureSub{i,2} = length(featureSub{i,4});
end
for i = 1:15
featureSub{i,3}=2626;
end
featureSub_all(:,7:12)=featureSub_69To2626;
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
featureSub{i,1}tmp_data.scoreResult(1,3);
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=find(ismember(featurei, tmp_data.featureSubset(1)));
featureSub{i,2} = length(featureSub{i,4});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
featureSub{i,1}=tmp_data.scoreResult(1,3);
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=find(ismember(featurei, tmp_data.featureSubset(1)));
featureSub{i,2} = length(featureSub{i,4});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
featureSub(:,7:12)=featureSub_69;
featureSub_all(:,7:12)=featureSub;
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
featureSub{i,1}=tmp_data.scoreResult(2,3);
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=find(ismember(featurei, tmp_data.featureSubset(1:2)));
featureSub{i,2} = length(featureSub{i,4});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
featureSub_all(:,7:12)=featureSub;
for i = 1:15
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
featureSub{i,1}=tmp_data.scoreResult(3,3);
featureSub{i,3}=length(tmp_data.featureSubset);
featureSub{i,4}=find(ismember(featurei, tmp_data.featureSubset(1:3)));
featureSub{i,2} = length(featureSub{i,4});
featureSub{i,5}=tmp_data.featureSubset;
featureSub{i,6}=tmp_data.scoreResult;
end
featureSub_all(:,7:12)=featureSub;
load('data_featureSelection_Species_0007.mat', 'scoreResult')
for i = 1:15
featureSub{i,1}=featureSub_all{i,4};
end
load('data_featureSelection_Species_0004.mat')
load('data_featureSelection_Species_0007.mat')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'featurei')
featureSub{7,1}=find(ismember(featurei, featureSubset(1:3)));
load('data_featureSelection_Species_0003.mat')
featureSub{3,1}=find(ismember(featurei, featureSubset(1:2)));
load('data_featureSelection_Species_0003.mat', 'scoreResult')
load('data_featureSelection_Species_0007.mat')
load('9.result_20130415_species1_15_data.mat', 'featureSubset_all')
load('1.fish7k2_feature_20130408_prefixNormalized.mat', 'class_id', 'features')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('9.result_20130415_species1_15_data.mat', 'class_id_15')
load('9.result_20130415_species1_15_data.mat', 'predict')
load('9.result_20130415_species1_15_data.mat', 'prior')
[ e20130415_models, e20130415_featureSub, e20130415_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130415 );
featureSub_20130415_revise_137
featureSub_20130415_revise_137= featureSub_20130415;
load('3.fish7k2_fs_flat_svm_fold5_individual_nonnormal.mat', 'scoreResult')
[mv,mi]=max(scoreResult(:,3))
load('3.fish7k2_fs_flat_svm_fold5_individual_nonnormal.mat', 'featureSubset')
featureSub_20130415_revise_137{1,1}=featureSubset(1:87);
featureSub_20130415_revise_137{3,1}=featureSubset(1:87);
featureSub_20130415_revise_137{7,1}=featureSubset(1:87);
featureSub_20130415_revise_137{9,1}=featureSubset(1:87);
[ e20130415Revise_models, e20130415Revise_featureSub, e20130415Revise_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130415 );
[ e20130415Revise_models, e20130415Revise_featureSub, e20130415Revise_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130415_revise_1379 );
[ e20130502_models, e20130502_featureSub, e20130502_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130502 );
load('3.fish7k2_fs_flat_svm_fold5_individual_nonnormal.mat', 'scoreResult')
load('3.fish7k2_fs_flat_svm_fold5_individual_nonnormal.mat', 'featureSubset')
featureSub_20130502{1,1}=featureSubset(1:87);
[ e20130502_models, e20130502_featureSub, e20130502_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130502 );
featureSub_20130415{1,1}=featureSubset(1:87);
[ e20130415_models, e20130415_featureSub, e20130415_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130415 );
[ e20130502_models, e20130502_featureSub, e20130502_componentNum ] = rejection_applyFS( fish7k2_features, fish7k2_class_id, featureSub_20130502 );
for i = 1:15
for j = 1:15
e20130415_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415_models{i,j});
e20130415Revise_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415Revise_models{i,j});
e20130502_scores{i,j}=classify_GMM_predict(fish27k_features, e20130502_models{i,j});
end
end
e20130415_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415_models{i,j});
e20130415Revise_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415Revise_models{i,j});
for i = 1:15
for j = 1:15
if ~isempty( e20130415_models{i,j}.indic)
e20130415_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415_models{i,j});
end
if ~isempty( e20130415Revise_models{i,j}.indic)
e20130415Revise_scores{i,j}=classify_GMM_predict(fish27k_features, e20130415Revise_models{i,j});
end
if ~isempty( e20130502_models{i,j}.indic)
e20130502_scores{i,j}=classify_GMM_predict(fish27k_features, e20130502_models{i,j});
end
end
end
for i = 1:15
for j = 1:15
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(37370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(37370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores = zeros(37370,1);
end
end
end
load('9.result_20130502_species1_15_data.mat', 'e20130415_scores', 'e20130415Revise_scores', 'e20130502_scores')
for i = 1:15
for j = 1:15
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(37370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(37370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores = zeros(37370,1);
end
end
end
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(37370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(37370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores = zeros(37370,1);
end
load('9.result_20130502_species1_15_data.mat', 'e20130415_scores', 'e20130415Revise_scores', 'e20130502_scores')
for i = 1:15
for j = 1:15
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(37370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(37370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores{i,j} = zeros(37370,1);
end
end
end
for i = 1:15
figure(1);
append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(2);
append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(3);
append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
pause();
end
for i = 1:15
figure(1);
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(1,i) ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
figure(2);
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(2,i) ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
figure(3);
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(3,i) ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
pause();
end
for i = 1:15
figure(1);
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
figure(2);
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
figure(3);
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
end
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(1);
for i = 1:14
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
figure(2);
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
figure(3);
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
end
figure(1);
for i = 1:14
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
subplot(3,14,i*3-2);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP, FP, 0 );
end
ay( TP, FP, 0 );
end
figure(1);
for i = 1:14
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
subplot(3,14,i*3-2);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
subplot(3,14,i*3-1);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
subplot(3,14,i*3);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
end
subplot(3,14,i*3-2);
plotroc(classid,outputs);
plotroc(classid{1,1},outputs{1,1});
help plotroc
for i = 1:14
[ TP, FP]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(1);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
[ TP, FP]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(2);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
[ TP, FP]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(3);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP, FP, 1 );
pause();
end
for i = 1:14
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(1);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(2);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(3);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(1);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(2);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(3);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(2);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i. 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP1, FP1]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP2, FP2]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP3, FP3]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP1, FP1, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP2, FP2, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP3, FP3, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:14
FPr(1,i)=sum(FP{1,i}<0.02
for i = 1:14
thres = 0.02;
FPr(1,i)=sum(FP{1,i}<thres);
TPr(1,i)=sum(TP{1,i}<thres);
for i = 1:14
for j = 1:3
thres = 0.02;
FPr(j,i)=sum(FP{j,i}<thres)/lenght(FP{j,i});
TPr(j,i)=sum(TP{j,i}<thres)/lenght(TP{j,i});
end
end
for i = 1:14
for j = 1:3
thres = 0.02;
FPr(j,i)=sum(FP{j,i}<thres)/length(FP{j,i});
TPr(j,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
for i = 1:14
for j = 1:3
thres = 0.02;
result(j*2-1,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result(j*2,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
for i = 1:14
for j = 1:3
thres = 0.02;
result(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
load('9.result_20130502_species1_15_data.mat', 'e20130502_featureSub')
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:14
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( log(e20130415_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( log(e20130415Revise_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( log(e20130502_scores{i,i}+eps), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
end
for i = 1:14
for j = 1:3
thres = 0.01;
result(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
for i=1:15
for i = 1:15
for j = 1:15
for i = 1:15
tmp_score = []j;
for i = 1:15
tmp_score = [];
for j = 1:15
20130415Revis
tmp_score(j,:)=e20130415Revise_scores{i,j};
for i = 1:15
tmp_score20130415 = [];
tmp_score20130415Revise = [];
tmp_score20130502 = [];
for j = 1:15
tmp_score20130415(j,:)=e20130415_scores{i,j};
tmp_score20130415Revise(j,:)=e20130415Revise_scores{i,j};
tmp_score20130502(j,:)=e20130502_scores{i,j};
end
s20130415_scores{i,1}=tmp_score20130415;
s20130415_scoresRevise{i,1}=tmp_score20130415Revise;
s20130502_scores{i,1}=tmp_score20130502;
end
load('9.result_20130502_species1_15_data.mat', 'e20130415_scores', 'e20130415Revise_scores', 'e20130502_scores')
for i = 1:15
for j = 1:15
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(27370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(27370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores = zeros(27370,1);
end
end
end
load('9.result_20130502_species1_15_data.mat', 'e20130415_scores', 'e20130415Revise_scores', 'e20130502_scores')
for i = 1:15
for j = 1:15
if isempty( e20130415_scores{i,j})
e20130415_scores{i,j} = zeros(27370,1);
end
if isempty( e20130415Revise_scores{i,j})
e20130415Revise_scores{i,j} = zeros(27370,1);
end
if isempty( e20130502_scores{i,j})
e20130502_scores{i,j} = zeros(27370,1);
end
end
end
for j = 1:15
20130415Revis
tmp_score(j,:)=e20130415Revise_scores{i,j};
for i = 1:15
tmp_score20130415 = [];
tmp_score20130415Revise = [];
tmp_score20130502 = [];
for j = 1:15
tmp_score20130415(j,:)=e20130415_scores{i,j};
tmp_score20130415Revise(j,:)=e20130415Revise_scores{i,j};
tmp_score20130502(j,:)=e20130502_scores{i,j};
end
s20130415_scores{i,1}=tmp_score20130415;
s20130415_scoresRevise{i,1}=tmp_score20130415Revise;
s20130502_scores{i,1}=tmp_score20130502;
end
for i = 1:15
tmp_score20130415 = [];
tmp_score20130415Revise = [];
tmp_score20130502 = [];
for j = 1:15
tmp_score20130415(j,:)=e20130415_scores{i,j};
tmp_score20130415Revise(j,:)=e20130415Revise_scores{i,j};
tmp_score20130502(j,:)=e20130502_scores{i,j};
end
s20130415_scores{i,1}=tmp_score20130415;
s20130415_scoresRevise{i,1}=tmp_score20130415Revise;
s20130502_scores{i,1}=tmp_score20130502;
end
for i = 1:15
s20130415_scoresRevise{i,1}=tmp_score20130415Revise;
s20130502_scores{i,1}=tmp_score20130502;
for i = 1:15
tmp_score20130415 = s20130415_scores{i,1};
tmp_score20130415Revise = s20130415_scoresRevise{i,1};
tmp_score20130502 = s20130502_scores{i,1};
for j = 1:27370
post__score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j)*prior));
post__score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j)*prior));
post__score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j)*prior));
end
end
load('9.result_20130415_species1_15_data.mat', 'predict')
for i = 1:15
tmp_score20130415 = s20130415_scores{i,1};
tmp_score20130415Revise = s20130415_scoresRevise{i,1};
tmp_score20130502 = s20130502_scores{i,1};
for j = 1:27370
post__score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j)*prior));
post__score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j)*prior));
post__score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j)*prior));
end
end
tmp_score20130415(:,j)
tmp_score20130415(:,j)*prior
for i = 1:15
tmp_score20130415 = s20130415_scores{i,1};
tmp_score20130415Revise = s20130415_scoresRevise{i,1};
tmp_score20130502 = s20130502_scores{i,1};
for j = 1:27370
post__score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j).*prior));
post__score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j).*prior));
post__score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j).*prior));
end
end
for i = 1:15
tmp_score20130415 = log(eps+s20130415_scores{i,1});
tmp_score20130415Revise = log(eps+s20130415_scoresRevise{i,1});
tmp_score20130502 = log(eps+s20130502_scores{i,1});
for j = 1:27370
post_log_score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j).*prior));
post_log_score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j).*prior));
post_log_score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j).*prior));
end
end
for i = 1:15
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( post__score20130415, fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( post__score20130415Revise, fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( post__score20130502, fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:15
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( post__score20130415(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( post__score20130415Revise(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( post__score20130502(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:15
for j = 1:3
thres = 0.01;
result(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
for i = 1:15
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( post_log_score20130415(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( post_log_score20130415Revise(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( post_log_score20130502(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:15
for j = 1:3
thres = 0.01;
result_post_log(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result_post_log(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
for i = 1:15
for j = 1:3
thres = 0.01;
result_post_log(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result_post_log(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
load('9.result_20130502_species1_15_data_add0.mat', 'e20130415_scores', 'e20130415Revise_scores', 'e20130502_scores')
for i = 1:15
tmp_score20130415 = [];
tmp_score20130415Revise = [];
tmp_score20130502 = [];
for j = 1:15
tmp_score20130415(j,:)=e20130415_scores{i,j};
tmp_score20130415Revise(j,:)=e20130415Revise_scores{i,j};
tmp_score20130502(j,:)=e20130502_scores{i,j};
end
s20130415_scores{i,1}=tmp_score20130415;
s20130415_scoresRevise{i,1}=tmp_score20130415Revise;
s20130502_scores{i,1}=tmp_score20130502;
end
for i = 1:15
tmp_score20130415 = log(eps+s20130415_scores{i,1})-log(eps);
tmp_score20130415Revise = log(eps+s20130415_scoresRevise{i,1})-log(eps);
tmp_score20130502 = log(eps+s20130502_scores{i,1})-log(eps);
for j = 1:27370
post_log_score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j).*prior));
post_log_score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j).*prior));
post_log_score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j).*prior));
end
end
load('9.result_20130415_species1_15_data.mat', 'predict')
for i = 1:15
tmp_score20130415 = log(eps+s20130415_scores{i,1})-log(eps);
tmp_score20130415Revise = log(eps+s20130415_scoresRevise{i,1})-log(eps);
tmp_score20130502 = log(eps+s20130502_scores{i,1})-log(eps);
for j = 1:27370
post_log_score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j).*prior));
post_log_score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j).*prior));
post_log_score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j).*prior));
end
end
load('9.result_20130415_species1_15_data.mat', 'prior')
for i = 1:15
tmp_score20130415 = log(eps+s20130415_scores{i,1})-log(eps);
tmp_score20130415Revise = log(eps+s20130415_scoresRevise{i,1})-log(eps);
tmp_score20130502 = log(eps+s20130502_scores{i,1})-log(eps);
for j = 1:27370
post_log_score20130415(i,j)=tmp_score20130415(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415(:,j).*prior));
post_log_score20130415Revise(i,j)=tmp_score20130415Revise(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130415Revise(:,j).*prior));
post_log_score20130502(i,j)=tmp_score20130502(predict(j),j)*prior(predict(j))/(eps+sum(tmp_score20130502(:,j).*prior));
end
end
for i = 1:15
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( post_log_score20130415(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( post_log_score20130415Revise(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( post_log_score20130502(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:15
fprintf('%d\n',i);
figure(1);
[ TP{1,i}, FP{1,i}]=append_resultEvaluate( post_log_score20130415(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(2);
[ TP{2,i}, FP{2,i}]=append_resultEvaluate( post_log_score20130415Revise(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(3);
[ TP{3,i}, FP{3,i}]=append_resultEvaluate( post_log_score20130502(i,:), fish27k_class_id_15, fish27k_predict,  i, 1);
figure(4);
[ auc(1,i), fpr,tpr,thresholds, classid{i,1},outputs{i,1} ] = append_CalcROCFromSeperateArray( TP{1,i}, FP{1,i}, 1 );
figure(5);
[ auc(2,i), fpr,tpr,thresholds, classid{i,2},outputs{i,2} ] = append_CalcROCFromSeperateArray( TP{2,i}, FP{2,i}, 1 );
figure(6);
[ auc(3,i), fpr,tpr,thresholds, classid{i,3},outputs{i,3} ] = append_CalcROCFromSeperateArray( TP{3,i}, FP{3,i}, 1 );
pause();
end
for i = 1:15
for j = 1:3
thres = 0.01;
result_post_log(j*2,i)=sum(FP{j,i}<thres)/length(FP{j,i});
result_post_log(j*2-1,i)=sum(TP{j,i}<thres)/length(TP{j,i});
end
end
load('9.result_20130502_species1_15_result.mat')
load('9.result_20130502_species1_15_data_add0.mat')
%-- 2013-05-02 05:21 PM --%
load('9.result_20130502_species1_15_result.mat')
result = [result_logScore;result_post;result_post_log];
result_logScore(1:6,15)=0;
result = [result_logScore;result_post;result_post_log];
%-- 2013-05-08 10:26 PM --%
folders = dir('.');
for i = 3:579
folders_name{i-1}=folders(i).name;
end
folders_name = folders_name';
for i = 3:580
folders_name{i-2}=folders(i).name;
end
for i = 3:580
folders_name{i-2,1}=folders(i).name;
end
index = 1;
for i = 1:580
files = dir(folders_name{i});
for j = 3:length(files)
files_name{i}{j-2,1}=files(j).name;
end
end
files_name{574}{1}
files_name{574}{1}(1)
files_name{574}{1}(1:5)
files_name{574}{1}(6:end)
for i = 1:578
for j = 1:length(files_name{i})
if files_name{i}{j}(1:5)=='image'
combines{index,1} = files_name{i};
combines{index,2} = files_name{i}{j};
combines{index,3} = ['mask', files_name{i}{j}(6:end)];
index = index + 1;
end
end
end
for i = 1:578
for j = 1:length(files_name{i})
if files_name{i}{j}(1:5)=='image'
combines{index,1} = folders_name{i};
combines{index,2} = files_name{i}{j};
combines{index,3} = ['mask', files_name{i}{j}(6:end)];
index = index + 1;
end
end
end
index = 1;
for i = 1:578
for j = 1:length(files_name{i})
if files_name{i}{j}(1:5)=='image'
combines{index,1} = folders_name{i};
combines{index,2} = files_name{i}{j};
combines{index,3} = ['mask', files_name{i}{j}(6:end)];
index = index + 1;
end
end
end
index = 1;
for i = 1:578
for j = 1:length(files_name{i})
if files_name{i}{j}(1:5)=='image'
combines{index,1} = folders_name{i};
combines{index,2} = files_name{i}{j};
combines{index,3} = ['mask', files_name{i}{j}(6:end)];
index = index + 1;
end
end
end
72322/3600
%-- 2013-05-08 10:55 PM --%
load('1.fish70k_attributes.mat')
for i = 36001:72322
fprintf('processing %d/72322\n', i);
image_fish{1} = imread([combines{i,1},'\',combines{i,2}]);
image_mask{1} = imread([combines{i,1},'\',combines{i,3}]);
features(i,:)= interface_generateFeatureSet(image_fish, image_mask);
end
load('matlab.mat')
features2(1:36000,:)=features;
load('data_15Species.mat', 'feature_mean', 'feature_std')
[features]= interface_normalizeFeatureFromPrefix(features, feature_mean, feature_std);
load('data_15Species.mat', 'BGOT_Trained')
[result_predict] = classify_HierSVM_predict(features, BGOT_Trained);
load('data_15Species.mat', 'convert2database')
[ result_predict_convert ] = append_convertlabel( convert2database, result_predict_15);
tabulate(result_predict_15)
for i = 1:72322
file_name{i,1}=fullfile('D:\Fish\2013-05-08_fish70K', combines{i,1}, combines{i,2});
end
for i = 1:72322
combines{i,4}=blured(i);
end
sum(blured)
for i = 1:72322
combines{i,5}=result_predict_convert(i);
end
for i = 1:72322
combines{i,5}=blured(i);
combines{i,4}=result_predict_convert(i);
end
[samples, dimens] = size(featureTest);
[sizeA, sizeB] = size(models);
sizeA = 15;
load('9.result_20130513_species1_15_TrainTest.mat', 'models')
for i = 1:15
indic = result_predict_15 == i;
rejection_score(indic,1) = classify_GMM_predict(features(indic,:), model{i,i});
end
for i = 1:15
indic = result_predict_15 == i;
rejection_score(indic,1) = classify_GMM_predict(features(indic,:), models{i,i});
end
rejection_score = log(rejection_score+eps)-log(eps);
rejection_indic = ismember(result_predict_15, [1,2,4,5,7,9,10,11,12,13,14]) & rejection_score < 2);
rejection_indic = ismember(result_predict_15, [1,2,4,5,7,9,10,11,12,13,14]) & (rejection_score < 2);
sum(rejection_indic)
tabulate(result_predict_15(rejection_indic))
combines(:,6)=rejection_indic;
for i = 1:72322
combines{i,6}=rejection_indic(i);
end
tabulate(result_predict_15)
rejection_indic1 = ismember(result_predict_15, [1,2,4,5,7,9,10,11,12,13,14]) & (rejection_score < 1);
tabulate(result_predict_15(rejection_indic1))
tabulate(result_predict_15(rejection_indic))
for i = 1:72322
combines{i,7}=rejection_score(i);
end
%-- 2013-05-14 03:45 PM --%
%-- 2013-05-20 01:56 PM --%
load('1.fish7k2_feature_20130205_slefNormalized95.mat')
help princomp
[COEFF, SCORE, LATENT] = princomp(features);
for i = 1:69
[COEFFi{i}, SCOREi{i}, LATENTi{i}] = princomp(features(:featurei == i));
for i = 1:69
[COEFFi{i}, SCOREi{i}, LATENTi{i}] = princomp(features(:,featurei == i));
end
sLATENT = cumsum(LATENT)/sum(LATENT);
for i = 1:69
sLATENTi{i}= cumsum(LATENTi{i})/sum(LATENTi{i});
end
for i = 1:69
topi(i,1)=find(sLATENTi{i}>0.8)(1);
end
for i = 1:69
tmp = find(sLATENTi{i}>0.8);
topi(i,1)=tmp(1)
end
for i = 1:69
featureSet{i}=SCOREi{i}(:,topi(i));
end
for i = 1:69
featureSet{i}=SCOREi{i}(:,1:topi(i));
end
for i = 1:69
featuresPCA = [];
for i = 1:69
featuresPCA = [featuresPCA;featureSet{i}];
end
for i = 1:69
featuresPCA = [featuresPCA,featureSet{i}];
end
featureiPCA = [];
for i = 1:69
featureiPCA(end+1,end+topi(i))=i;
end
featureiPCA = [];
for i = 1:69
featureiPCA(end+1,end+topi(i),1)=i;
end
featureiPCA = [];
for i = 1:69
featureiPCA(end+1:end+topi(i),1)=i;
end
load('0.fish7k2_attribute.mat', 'cv_id')
[featureSubset, scoreResult] = classify_featureselection_fw(featuresPCA, featureiPCA, class_id, traj_id, 69, 3, cv_id, 2, 'pca_fw', 0);
max(scoreResult)
for i = 1:69
tmp = find(sLATENTi{i}>0.8);
top80(i,1)=tmp(1)
end
for i = 1:69
tmp = find(sLATENTi{i}>0.8);
top80(i,1)=tmp(1);
tmp = find(sLATENTi{i}>0.9);
top90(i,1)=tmp(1);
tmp = find(sLATENTi{i}>0.95);
top95(i,1)=tmp(1);
end
featureiPCA80 = [];
featureiPCA90 = [];
featureiPCA95 = [];
for i = 1:69
featureiPCA80(end+1:end+top80(i),1)=i;
featureiPCA90(end+1:end+top90(i),1)=i;
featureiPCA95(end+1:end+top95(i),1)=i;
end
featuresPCA80 = [];
featuresPCA90 = [];
featuresPCA95 = [];
for i = 1:69
featuresPCA80 = [featuresPCA80;SCOREi{i}(:,1:top80(i))];
featuresPCA90 = [featuresPCA90;SCOREi{i}(:,1:top90(i))];
featuresPCA95 = [featuresPCA95;SCOREi{i}(:,1:top95(i))];
end
featuresPCA80 = [];
featuresPCA90 = [];
featuresPCA95 = [];
for i = 1:69
featuresPCA80 = [featuresPCA80,SCOREi{i}(:,1:top80(i))];
featuresPCA90 = [featuresPCA90,SCOREi{i}(:,1:top90(i))];
featuresPCA95 = [featuresPCA95,SCOREi{i}(:,1:top95(i))];
end
[featureSubset80, scoreResult80] = classify_featureselection_fw(featuresPCA80, featureiPCA80, class_id, traj_id, 69, 3, cv_id, 2, 'pca_fw80', 0);
[featureSubset90, scoreResult90] = classify_featureselection_fw(featuresPCA90, featureiPCA90, class_id, traj_id, 69, 3, cv_id, 2, 'pca_fw90', 0);
[featureSubset95, scoreResult95] = classify_featureselection_fw(featuresPCA95, featureiPCA95, class_id, traj_id, 69, 3, cv_id, 2, 'pca_fw95', 0);
[featureSubset80, scoreResult80] = classify_featureselection_fw(featuresPCA80, featureiPCA80, class_id, traj_id, 69, 3, cv_id, 2, 'pca_fw80', 0);
%-- 2013-05-24 10:56 AM --%
load('0.fish7k2_attribute.mat', 'cv_id')
script_HPC_NodeSplit
script_HPC_NodeSplit(1,1)
start_time = clock();
end\_time = clock();
end_time = clock();
%-- 2013-06-02 11:00 PM --%
help videoreader
OBJ = VideoReader('gt_Video2_cut_1.avi');
vidFrames = read(OBJ);
help read
vidFrames = read(OBJ, 1:1000);
vidFrames = read(OBJ, [1 1000]);
size(vidFrames)
vidFrames = read(OBJ, [1 10]);
imshow(vidFrames(:,:,:,1))
imshow(vidFrames(:,:,:,9))
dmnode = xmlread('video2_cut1.xml');
dmstruct = parseChildNode(dmnode);
dmnode = parseXML('video2_cut1.xml');
dmnode = parseXML('video2_cut2.xml');
load('video2_cut1.mat')
help polygon_string
help read
help videoreader
load('video2_cut_xml.mat')
[ fish_image, mask_image ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
ceil(numFrames/frame_block)
imshow(fish_image{1})
imshow(mask_image{1})
[ fish_image, mask_image ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
imshow(fish_image{1})
imshow(mask_image{1})
[ fish_image_cut1, mask_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
for i = 1:4840
load('video2_cut_xml.mat', 'detects_xml_cut1_num')
[ fish_image_cut1, mask_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
imshow(draw_image{1})
[ fish_image_cut1, mask_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
imshow(draw_image{1})
[ fish_image_cut1, mask_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
imshow(draw_image{1})
[ fish_image_cut1, mask_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
imshow(draw_image{1})
[ fish_image_cut1, mask_image_cut1, draw_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
for = 1:100
for i = 1:100
imshow(draw_image_cut1{i});
pause();
end
[ fish_image_cut1, mask_image_cut1, draw_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
for i = 1:100
imshow(draw_image_cut1{i});
pause();
end
[ fish_image_cut1, mask_image_cut1, draw_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
for i = 1:4840
imwrite(draw_image_cut1{i}, sprintf('fish_%04d.png', i));
end
[features_cut1_raw]= interface_generateFeatureSet(fish_image_cut1, mask_image_cut1);
%-- 2013-06-24 02:39 PM --%
load('1.fish70k_attributes.mat')
load('2.fish70k_predict.mat', 'combines')
prediction = [combines_a(:,4);];
prediction = [combines_a{:,4};];
prediction = [combines_a{:,4},];
prediction = [combines_a{:,4};];
prediction = prediction';
tabulate(prediction)
clown_indic = prediction == 9;
attributes = combines(clown_indic, :);
unique(attributes(:,1))
video_id=unique(attributes(:,1));
video_id=unique(prediction_clown(:,1));
videolist.video_id =video_id(201:end);
script_downloadVideo_fromGTInfo
clear;
load('1.fish_id.mat', 'clown_fish_id_unique')
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT * FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9)'];
[detection_detection_id, detection_fish_id, detection_video_id,  detection_frame_id, detection_time_stamp, detection_bb_cc, detection_detection_certainty, detection_tracking_certainty, detection_component_id] = mysql(sql);
mysql( 'close');
mysql( 'open', host, user, password);
mex config
mex mysql
mex mysql.m
mex mysql.cpp
mex setup
mex -setup
help mex
mex -I'C:\Users\Phoenix\Desktop\mysql-5.6.12\include' mysql.cpp
mex -I'C:\Users\Phoenix\Desktop\mysql-5.6.12\include' mysql
mex -I'C:\Users\Phoenix\Desktop\mysql-5.6.12\include' mysql.m
mex -I'C:\Users\Phoenix\Desktop\mysql-5.6.12\include' mysql.cpp
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' mysql.cpp
help mex
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib' mysql.cpp
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmySQL.lib' mysql.cpp
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib' mysql.cpp
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib' -DWIN32 mysql.cpp 'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib'
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib' mysql.cpp 'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib'
mysql( 'open', host, user, password);
mex -I'C:\Program Files\MySQL\MySQL Server 5.6\include' -L'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib' -DWIN32 mysql.cpp 'C:\Program Files\MySQL\MySQL Server 5.6\lib\libmysql.lib'
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT * FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9)'];
[detection_detection_id, detection_fish_id, detection_video_id,  detection_frame_id, detection_time_stamp, detection_bb_cc, detection_detection_certainty, detection_tracking_certainty, detection_component_id] = mysql(sql);
%-- 2013-06-26 01:35 PM --%
load('1.fish_id.mat', 'clown_fish_id_unique')
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT video_id FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9)'];
[detection_video_id] = mysql(sql);
%-- 2013-06-26 02:11 PM --%
%-- 2013-06-28 05:25 PM --%
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'featurei')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
[hier_arch] = classify_HierSVM_train(features, class_id, featurei, hier_tree_1301_69a, 2);
hier_arch__1301_69f_light = hier_arch__1301_69f;
load('data_15Species.mat', 'feature_mean', 'feature_std', 'convert2database')
load('data_10Species.mat', 'feature_mean')
load('data_15Species.mat')
sample_predictSpecies
clear;
load('data_23Species.mat')
feature_mean = feature_mean';
feature_std = feature_std';
sample_predictSpecies
clear;
load('data_23Species.mat')
convert2database= 1:23;
convert2database= 1:23';
convert2database= (1:23)';
clear;
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'featurei', 'features')
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
[ models ] = rejection_applyFS( features, class_id, featureSub_20130415Revise );
load('9.result_20130529_species1_15_BGOT_data.mat', 'class_id_15')
load('9.result_20130529_species1_15_BGOT_data.mat', 'prior')
indic = class_id_15 > 0 & class_id_15 <=6;
class_id = class_id_15(indic);
features = features(indic, :);
featureSub = featureSub_20130415Revise(1:6);
[ models ] = rejection_applyFS( features, class_id, featureSub_20130415Revise );
[ models ] = rejection_applyFS( features, class_id, featureSub);
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
[ models ] = rejection_applyFS( features, class_id, featureSub_20130415Revise );
[ models ] = rejection_applyFS( features, class_id_15, featureSub_20130415Revise );
[ models ] = rejection_applyFS( features(class_id_15>0,:), class_id_15(class_id_15>0,:), featureSub_20130415Revise );
load('1.BGOT_evaluate.mat', 'result_predict')
load('9.result_20130530_NIPS_result.mat', 'predict')
prediction = predict(:,1);
[ post_log_score, scores_log_diag ] = classify_rejectionGMM( models, features, prediction, prior );
sum(post_log_score> 0.01)
%-- 2013-06-29 10:01 PM --%
sample_predictSpecies
%-- 2013-06-29 11:03 PM --%
sample_predictSpecies
imshow(input_rgbImage{4})
load('1.BGOT_evaluate.mat', 'result_predict')
load('1.BGOT_evaluate.mat', 'result_score')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
[ result_predict_gmm, result_score_gmm ] = interface_rejection( result_predict, result_score, features, model_data );
tabulate(prediction_15)
[ result_predict_gmm, result_score_gmm ] = interface_rejection( result_predict, result_score, features, model_data );
load('9.result_20130529_species1_15_BGOT_data.mat', 'class_id_15')
class_id_15 = class_id_15(rejection_indic);
prediction_15 = prediction_15(rejection_indic);
sum(rejection_reject)
result_predict2 = model_data.convert2database(result_predict, 2);
tabulate(result_predict);
tabulate(result_predict2);
clear;
load('2.GMM_evaluate.mat', 'prediction_15', 'post_log_score', 'class_id_15')
for i = 1:6
TP = class_id_15 == i & prediction_15 == i;
FP = class_id_15 ~= i & prediction_15 == i;
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
sum(TP<0.01)
for i = 1:6
TP = post_log_score(class_id_15 == i & prediction_15 == i);
FP = post_log_score(class_id_15 ~= i & prediction_15 == i);
subplot(2,2,1);
title('TP');
plot(histc(TP, -1:0.01:1)/length(TP));
subplot(2,2,2);
title('FP');
plot(histc(FP, -1:0.01:1)/length(FP));
pause();
end
sum(TP<0.01)
load('9.result_20130529_species1_15_BGOT_data.mat', 'class_id_15')
load('9.result_20130529_species1_15_BGOT_data.mat', 'prior')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'featurei')
find(class_id_15)=5;
find(class_id_15==5);
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
[ models, featureSub, componentNum ] = rejection_applyFS( feature_train, classid_train, featureSub_20130415Revise{5} );
[ models, featureSub, componentNum ] = rejection_applyFS( features, class_id, featureSub_20130415Revise );
[ models, featureSub, componentNum ] = rejection_applyFS( features, class_id_15, featureSub_20130415Revise );
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0), class_id_15(class_id_15>0), featureSub_20130415Revise );
i = 5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0,:), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0,:), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0,:), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
[ models, featureSub, componentNum ] = rejection_applyFS( features(class_id_15>0,:), class_id_15(class_id_15>0), featureSub_20130415Revise );
i=5;
load('matlab.mat', 'models')
[ prediction, scores ] = interface_rejection( class_id_15(class_id_15==5), 0, features(class_id_15==5,:), models )
load('data_23Species.mat', 'GMM_prior')
[ post_log_score ] = classify_rejectionGMM( models, features(rejection_indic, :), prediction_15(rejection_indic), GMM_prior );
[ post_log_score ] = classify_rejectionGMM( models, features(class_id_15==5, :), class_id_15(class_id_15==5), GMM_prior );
i = 5;
break;
i = 16;
[ post_log_score ] = classify_rejectionGMM( models, features(class_id_15==5, :), class_id_15(class_id_15==5), GMM_prior );
clear;
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
if iscell(result_scores)
result_scores = cell2mat(result_scores);
end
result_score = zeros(length(result_predict), 1);
for i = 1:length(result_predict)
result_score(i,1) = result_scores(i, result_predict(i));
end
[ result_predict, result_score ] = interface_rejection( result_predict, result_score, features, model_data );
if exist('traj_id', 'var')
result_predict =  result_trajvote( result_predict, traj_id, result_score);
end
result_predict = model_data.convert2database(result_predict, 2);
clear;
load('images.mat')
addpath('./fish_recog');
tmp_data = load('image.mat');
[ prediction, scores ] = interface_recognizeFishFromImage(tmp_data.model_data , fish_image_cut1,mask_image_cut1,  traj_id);
[ prediction1, scores1 ] = interface_recognizeFishFromImage(tmp_data.model_data , fish_image_cut1,mask_image_cut1);
[ prediction2, scores2 ] = interface_recognizeFishFromImage(tmp_data.model_data , fish_image_cut2,mask_image_cut2);
[ prediction1, scores1 ] = interface_recognizeFishFromImage(model_data , fish_image_cut1,mask_image_cut1);
[ prediction2, scores2 ] = interface_recognizeFishFromImage(model_data , fish_image_cut2,mask_image_cut2);
model_data = load('data_23Species.mat');
[ prediction1, scores1 ] = interface_recognizeFishFromImage(tmp_data.model_data , fish_image_cut1,mask_image_cut1);
[ prediction1, scores1 ] = interface_recognizeFishFromImage(model_data , fish_image_cut1,mask_image_cut1);
%-- 2013-07-02 04:30 PM --%
load('images.mat', 'fish_image_cut2', 'mask_image_cut2')
addpath('./fish_recog');
model_data = load('data_23Species.mat');
[ prediction2, scores2 ] = interface_recognizeFishFromImage(model_data , fish_image_cut2,mask_image_cut2);
%-- 2013-07-02 05:10 PM --%
load('1.fish_id.mat', 'clown_fish_id_unique')
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT * FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) limit 0,1000000'];
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) limit 0,1000000'];
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) limit 0,2000000'];
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) limit 0,6000000'];
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
length(unique(detection_fish_id1))
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) group by (`fish_id`) limit 0,6000000'];
length(unique(detection_fish_id1))
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
%-- 2013-07-02 06:57 PM --%
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) group by (`fish_id`) limit 0,100000'];
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
unique(detection_fish_id1)
mysql( 'close');
clear;
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT * FROM  `fish_detection_p01` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) group by (`fish_id`)'];
[detection_detection_id1, detection_fish_id1, detection_video_id1,  detection_frame_id1, detection_time_stamp1, detection_bb_cc1, detection_detection_certainty1, detection_tracking_certainty1, detection_component_id1] = mysql(sql);
mysql( 'close');
clear;
cls;
%-- 2013-07-03 12:09 PM --%
load('result_20130610_species15.mat', 'result_predict_cut1', 'result_predict_cut2')
sum(result_predict_cut1==0)
load('result_20130610_species15.mat', 'rejection_cut1', 'rejection_cut2')
sum(rejection_cut1)
sum(rejection_cut2)
load('result_20130702_species23.mat', 'prediction1', 'scores1', 'prediction2', 'scores2')
prediction1(rejection_cut1==1)=0;
scores1(rejection_cut1==1)=0;
scores2(rejection_cut2==1)=0;
prediction2(rejection_cut2==1)=0;
load('images.mat', 'mask_image_cut1', 'mask_image_cut2')
load('images.mat', 'whole_image_cut1', 'whole_image_cut2')
for i = 1:4840
imwrite(whole_image_cut1{i}, sprintf('%02d_%.4f_%04d.png', prediction1(i), socres1(i), i));
end
for i = 1:4840
imwrite(whole_image_cut1{i}, sprintf('%02d_%.4f_%04d.png', prediction1(i), scores1(i), i));
end
tabulate(prediction1)
clear;
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT detection_id,fish_id,video_id,frame_id,timestamp,asText(bb_cc),detection_certainty,tracking_certainty,component_id FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) group by (`fish_id`) limit 0,1000000'];
[detection_detection_id, detection_fish_id, detection_video_id,  detection_frame_id, detection_time_stamp, detection_bb_cc, detection_detection_certainty, detection_tracking_certainty, detection_component_id] = mysql(sql);
mysql( 'close');
char(detection_bb_cc{1})
clear;
load('3.detection_top1m.mat', 'detection_bb_cc1')
detection_bb_cc1{1}
uint8(detection_bb_cc1{1})
char(uint8(detection_bb_cc1{1}))
coords = polygon_string(char(uint8(detection_bb_cc1{1})))
%-- 2013-07-05 03:38 PM --%
load('3.detection_top1m.mat', 'detection_bb_cc1')
for i = 500001:750000
i
[detection_bounding_box(i), detection_contour{i}]=get_shape_from_database_record(detection_bb_cc1{i});
end
user = 'f4kuser';
password = 'hunt4fish';
dbName = 'f4k_db';
host = 'gleoncentral.nchc.org.tw';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);
sql = ['SELECT detection_id,fish_id,component_id FROM  `fish_detection` WHERE  `fish_id` in (SELECT fish_id FROM  `traj_species` WHERE  `specie_id`=9) group by (`fish_id`) limit 0,1000000'];
[detection_detection_id, detection_fish_id, detection_component_id] = mysql(sql);
mysql( 'close');
load('3.detection_top1m.mat')
sum(detections.fish_id==detection_fish_id)
detections.detection_id = detection_detection_id;
detections.component_id = detection_component_id;
clear;
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'featurei')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
result_scores = cell2mat(result_scores);
load('1.BGOT_evaluate.mat', 'result_predict')
sum(result_predict==result_predict)
load('1.BGOT_evaluate.mat', 'result_score')
load('1.BGOT_evaluate.mat', 'result_scores_23')
result_score2 = zeros(length(result_predict), 1);
for i = 1:length(result_predict)
result_score2(i,1) = result_scores_23(i, result_predict(i));
end
sum(result_score2==result_score)
find(result_score2~=result_score)
model_data = load('data_23Species.mat');
[result_predict, result_score ] = interface_rejection( result_predict, result_score, features, model_data );
load('3.BGOT_evaluate_fromUnormalized.mat', 'result_predict', 'result_score', 'result_scores')
for i = 1:6
[result_predict_i{i,1}, result_score_i{i,1} ] = interface_rejection( result_predict, result_score, features, model_data,i );
end
[result_predict_i{i,1}, result_score_i{i,1} ] = interface_rejection( result_predict, result_score, features, model_data,i );
for i = 1:6
[result_predict_all{i,1}, result_score_all{i,1} ] = interface_rejection( result_predict, result_score, features, model_data,i );
end
for i = 1:6
sum(result_score_all{i}, result_score_i{i})
end
for i = 1:6
sum(result_score_all{i}==result_score_i{i})
end
for i = 1:6
[result_predict_all{i,1}, result_score_all{i,1} ] = interface_rejection( result_predict, result_score, features, model_data,i );
end
for i = 1:6
[result_predict_all{i,1}, result_score_all{i,1} ] = interface_rejection( result_predict, result_score, features, model_data,i );
end
for i = 1:6
sum(result_score_all{i}==result_score_i{i})
end
for i = 1:6
find(result_score_all{i}~=result_score_i{i})
end
load('3.BGOT_evaluate_fromUnormalized.mat', 'result_predict', 'result_score')
[ prediction, scores ] = interface_rejection( prediction, scores, features, model_data );
[ prediction, scores ] = interface_rejection( result_predict, result_score, features, model_data );
%-- 2013-07-07 10:48 PM --%
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'featurei')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'traj_id')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features(1,:), model_data.BGOT_1301_69f);
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features(1,:), model_data.BGOT_1301_69f);
[result_predict, result_scores] = classify_HierSVM_predict(features(1,:10,:), model_data.BGOT_1301_69f);
if iscell(result_scores)
result_scores = cell2mat(result_scores);
end
[result_predict, result_scores] = classify_HierSVM_predict(features(1:10,:), model_data.BGOT_1301_69f);
if iscell(result_scores)
result_scores = cell2mat(result_scores);
end
%-- 2013-07-07 10:53 PM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
rt_scores = append_hierScoreToFlat(hier_arch, rt_scores);
rt_scores = cell2mat(rt_scores);
indic = rt_predict ~= -1;
rt_predict1 = rt_predict(indic);
rt_scores = rt_scores(indic,:);
[mv,mi]=max(rt_scores');
mi = mi;
mi = mi;;
mi = mi';
sum(mi == rt_predict)
sum(mi == rt_predict1)
tabulate(mi)
load('1.fish27k_attributes.mat', 'class_id')
class_id = class_id(indic);
result_evaluate(mi, class_id);
result_evaluate(mi, class_id)
tabulate(class_id)
clear;
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
result_scores = cell2mat(result_scores);
[mv,mi]=max(result_scores');
mi = mi';
sum(mi(1:27370),result_predict)
sum(mi(1:27370)==result_predict)
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
[result_predict, result_scores] = classify_HierSVM_predict(features(1:10), model_data.BGOT_1301_69f);
[result_predict, result_scores] = classify_HierSVM_predict(features(1:10,:), model_data.BGOT_1301_69f);
mat2cell(tmp_scores(:, tmp_order))
[result_predict, result_scores] = classify_HierSVM_predict(features(1:10,:), model_data.BGOT_1301_69f);
mat2cell(tmp_scores(:, tmp_order), ones(size(tmp_scores, 1),1), size(tmp_scores, 2))
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
%-- 2013-07-08 01:02 AM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
[mv,mi]=mat2cell(result_scores);
[mv,mi]=max(mat2cell(result_scores));
result_allscores = cell2mat(result_scores);
[mv,mi]=max(result_allscores);
[mv,mi]=max(result_allscores, [], 2);
sum(mi==result_predict)
find(mi~=result_predict)
result_allscores(mi~=result_predict,:)
socre_=result_allscores(mi~=result_predict,:);
result_predict(mi~=result_predict,:)
if iscell(result_scores)
result_scores = cell2mat(result_scores);
end
result_score = zeros(length(result_predict), 1);
for i = 1:length(result_predict)
result_score(i,1) = result_scores(i, result_predict(i));
end
[GMM_result_predict, GMM_result_score, GMM_result_score_rejection ] = interface_rejection( result_predict, result_score, features, model_data );
clear;
load('7.BGOT_evaluate_fromUnormalized_20130708.mat')
indic = GMM_result_predict == 0;
BGOT_result_scores(indic,:)=0;
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'class_id', 'traj_id')
load('7.evaluate_fromUnormalized_20130708.mat', 'BGOT_result_predict')
result_evaluate(BGOT_result_predict, class_id)
load('1.BGOT_evaluate.mat', 'result_predict')
sum(BGOT_result_predict == result_predict)
load('3.BGOT_evaluate_fromUnormalized.mat', 'result_predict')
sum(BGOT_result_predict == result_predict)
find(BGOT_result_predict ~= result_predict)
result_evaluate(BGOT_result_predict, class_id)
clear;
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features(1,:), model_data.BGOT_1301_69f);
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features(1,:), model_data.BGOT_1301_69f);
%-- 2013-07-09 11:44 AM --%
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'class_id')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'featurei')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'traj_id')
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
featureSub(1:23) = featureSub(1);
tmp_sub = randperm(67);
featureSub(11:23) = featureSub(1){tmp_sub(1:10)};
featureSub{11:23} = featureSub(1){tmp_sub(1:10)};
tmp_subFeature = featureSub{1}({tmp_sub(1:10));
tmp_subFeature = featureSub{1}(tmp_sub(1:10));
featureSub{11:23} = tmp_subFeature;
featureSub(11:23) = tmp_subFeature;
featureSub(11:23) = {tmp_subFeature};
[ models, featureSub, componentNum ] = rejection_applyFS( features, class_id, featureSubset );
[ models, featureSub, componentNum ] = rejection_applyFS( features, class_id, featureSub );
prior = tabulate(class_id);
prior = prior(:,2)/sum(prior(:,2));
for i = 1:23
[ post_log_score(i,:) ] = classify_rejectionGMM( models, features, i*ones(27370,1), prior );
end
for i = 1:23
[ post_log_score{i,:} ] = classify_rejectionGMM( models, features, i*ones(27370,1), prior );
end
for i = 1:23
[ post_log_score{i,:} ] = classify_rejectionGMM( models, features, i*ones(27370,1), prior );
end
post_log_score = {};
for i = 1:23
[ post_log_score{i,:} ] = classify_rejectionGMM( models, features, i*ones(27370,1), prior );
end
for i = 1:23
post_log_score(i) = {classify_rejectionGMM( models, features, i*ones(27370,1), prior )};
end
%-- 2013-07-09 02:35 PM --%
mmreader('21300584e7d5228d9db1d81b4ba9cc57#201001031010.flv')
%-- 2013-07-09 10:22 PM --%
load('matlab1.mat')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
post_log_score(i) = {classify_rejectionGMM( models, features, i*ones(27370,1), prior )};
i = 1;
post_log_score(i) = {classify_rejectionGMM( models, features, i*ones(27370,1), prior )};
for i = 1:23
post_log_score(i) = {classify_rejectionGMM( models, features, i*ones(27370,1), prior )};
end
%-- 2013-07-09 10:31 PM --%
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('matlab2.mat')
for i = 1:23
post_log_score(i) = {classify_rejectionGMM( models, features, i*ones(27370,1), prior )};
end
post_log_score_all = cell2mat(post_log_score);
%-- 2013-07-09 11:08 PM --%
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features', 'class_id', 'featurei', 'cv_id_fold5', 'traj_id')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id2=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id2=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id2(i)]);
end
sum(train_id)
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id2=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id2(i)]);
[hier_arch{i}] = classify_HierSVM_train(features(train_id,:), class_id(train_id,:), featurei, hier_tree_1301_69a, 2);
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id2=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id2(i)]);
[hier_arch{i}] = classify_HierSVM_train(features(train_id,:), class_id(train_id,:), featurei, hier_tree_1301_69a, 2);
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
[hier_arch{i}] = classify_HierSVM_train(features(train_id,:), class_id(train_id,:), featurei, hier_tree_1301_69a, 2);
end
for i = 3:5
train_id = ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
[hier_arch{i}] = classify_HierSVM_train(features(train_id,:), class_id(train_id,:), featurei, hier_tree_1301_69a, 2);
end
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
[result_predict{i}, result_scores{i}] = classify_HierSVM_predict(features(test_ids, :), hier_arch{i});
end
for i = 1:5
result_scores{i} = cell2mat(result_scores{i});
end
load('GMM_cv5.mat', 'models')
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
for j = 1:6
[GMM_result_predict{i,j}, GMM_result_score{i,j}, GMM_result_score_rejection{i,j} ] = interface_rejection( result_predict{i}, result_score{i}, features(test_ids, :), models{i}, j );
end
end
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
for j = 1:6
[GMM_result_predict{i,j}, GMM_result_score{i,j}, GMM_result_score_rejection{i,j} ] = interface_rejection( result_predict{i}, result_score{i}, features(test_ids, :), models{i}, j );
end
end
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
for j = 1:6
[GMM_result_predict{i,j}, GMM_result_score{i,j}, GMM_result_score_rejection{i,j} ] = interface_rejection( result_predict{i}, result_scores{i}, features(test_ids, :), models{i}, j );
end
end
%-- 2013-07-10 11:56 PM --%
%-- 2013-07-11 12:02 AM --%
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features', 'class_id', 'featurei', 'cv_id_fold5', 'traj_id')
load('BGOT_cv5.mat', 'hier_arch')
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
[BGOT_result_predict{i}, BGOT_result_scores{i}] = classify_HierSVM_predict(features(test_ids, :), hier_arch{i});
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
[BGOT_result_predict{i}, BGOT_result_scores{i}] = classify_HierSVM_predict(features(test_ids, :), hier_arch{i});
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
[BGOT_result_predict{i}, BGOT_result_scores{i}] = classify_HierSVM_predict(features(test_ids, :), hier_arch{i});
end
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
test_ids = cv_id_fold5 == test_id(i);
[BGOT_result_predict{i}, BGOT_result_scores{i}] = classify_HierSVM_predict(features(test_ids, :), hier_arch{i});
BGOT_result_scores{i} = cell2mat(BGOT_result_scores{i});
end
%-- 2013-07-11 12:05 AM --%
load('9.result_20130529_species1_15_BGOT_data.mat', 'class_id_15')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'cv_id_fold5', 'features')
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 1:5
train_id = class_id_15 > 0 & ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
class_id_cv = tabulate(class_id_15(train_id));
prior{i,1}=class_id_cv(:,2)/sum(class_id_cv(:,2));
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
for i = 1:5
train_id = class_id_15 > 0 & ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
class_id_cv = tabulate(class_id_15(train_id));
prior{i,1}=class_id_cv(:,2)/sum(class_id_cv(:,2));
end
for i = 1:3
train_id = class_id_15 > 0 & ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
class_id_cv = tabulate(class_id_15(train_id));
models{i,1}.prior=class_id_cv(:,2)/sum(class_id_cv(:,2));
[ models{i,1}.GMM_models, models{i,1}.featureSub, models{i,1}.componentNum ] = rejection_applyFS( features(train_id,:), class_id_15(train_id), featureSubset );
end
%-- 2013-07-11 12:11 AM --%
load('9.result_20130529_species1_15_BGOT_data.mat', 'class_id_15')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'cv_id_fold5', 'features')
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'cv_id_fold5', 'features')
train_id1 = [1,2,3,4,5]; train_id2 = [2,3,4,5,1];train_id3=[3,4,5,1,2];test_id = [5,1,2,3,4];
for i = 4:5
train_id = class_id_15 > 0 & ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
class_id_cv = tabulate(class_id_15(train_id));
models{i,1}.prior=class_id_cv(:,2)/sum(class_id_cv(:,2));
[ models{i,1}.GMM_models, models{i,1}.featureSub, models{i,1}.componentNum ] = rejection_applyFS( features(train_id,:), class_id_15(train_id), featureSubset );
end
for i = 4:5
train_id = class_id_15 > 0 & ismember(cv_id_fold5, [train_id1(i),train_id2(i),train_id3(i)]);
class_id_cv = tabulate(class_id_15(train_id));
models{i,1}.prior=class_id_cv(:,2)/sum(class_id_cv(:,2));
[ models{i,1}.GMM_models, models{i,1}.featureSub, models{i,1}.componentNum ] = rejection_applyFS( features(train_id,:), class_id_15(train_id), featureSubset );
end
load('GMM_cv5.mat', 'models')
GMM_models(1:3)=models;
%-- 2013-07-11 01:53 AM --%
load('GMM_result_cv5.mat', 'traj_scores')
load('7.result_fromUnormalized_20130708.mat', 'trajs_classid_vote')
GMM_models(1:3)=models;
draw_result_compare6
sum(traj_scores{j,1}')>0
sum(traj_scores{j,1},2)>0
draw_result_compare6
traj_scores{j,1}(trajs_classid_vote==i&predict_1{j}==i,i)
trajs_classid_vote==i
find(trajs_classid_vote==i)
find(predict_1{j}==i)
%-- 2013-07-11 05:25 PM --%
load('2.fish90k_fishImage.mat')
load('1.fish90k_attributes_20121121.mat', 'class_id')
maskImg2 = maskImg(class_id==21)
for i = 1:14
imshow(maskImage2{i});
pause();
end
for i = 1:14
imshow(maskImg2{i});
pause();
end
%-- 2013-07-11 07:53 PM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'class_id')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'traj_id')
addpath('./fish_recog');
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[BGOT_result_predict, BGOT_result_scores_23] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
if iscell(BGOT_result_scores_23)
BGOT_result_scores_23 = cell2mat(BGOT_result_scores_23);
end
load('7.evaluate_fromUnormalized_20130708.mat', 'BGOT_result_predict', 'BGOT_result_scores')
load('evaluate.mat', 'BGOT_result_predict', 'BGOT_result_scores_23')
sum(BGOT_result_predict==BGOT_result_predict2)
sum(BGOT_result_scores2==BGOT_result_scores_23)
load('evaluate.mat', 'BGOT_result_scores_23')
load('evaluate.mat', 'BGOT_result_predict')
load('7.evaluate_fromUnormalized_20130708.mat', 'GMM_result_predict')
[GMM_result_predict, GMM_result_score ] = interface_rejection( BGOT_result_predict, BGOT_result_scores_23, features, model_data );
sum(GMM_result_predict==GMM_result_predict2)
sum(GMM_result_score(GMM_result_predict>0,:)==BGOT_result_scores_23(GMM_result_predict>0,:))
sum(GMM_result_score(GMM_result_predict==0,:))
sum(GMM_result_predict == 0)
[traj_predict, traj_scores] = result_trajvote_byScore( GMM_result_score, traj_id);
result_traj_id(i)
for i = 1:8663
traj_predict_single(i,1)=traj_predict{i}(1);
end
load('8.result_fromUnormalized_20130709.mat', 'traj_scores_b1')
[mv,mi]=predict_b1 = max(traj_scores_b1);
[mv,predict_b1] = max(traj_scores_b1);
[mv,predict_b1] = max(traj_scores_b1,2);
[mv,predict_b1] = max(traj_scores_b1,[],2);
sum(traj_predict_single==0)
sum(traj_predict_single>=0)
sum(traj_predict_single==predict_b1)
sum(traj_predict_single>=0)
traj_scores_b1(traj_predict_single==0,:)
sum(traj_scores_b1(traj_predict_single==0,:))
compare2 = traj_predict_single(traj_predict_single>0);
compare2(:,2) = predict_b1(traj_predict_single>0);
find(compare2(:,1)~=compare2(:,2))
sum(traj_predict(traj_predict_single>0)==traj_predict_single(traj_predict_single>0))
sum(traj_predict_b1(traj_predict_single>0)==traj_predict_single(traj_predict_single>0))
sum(predict_b1(traj_predict_single>0)==traj_predict_single(traj_predict_single>0))
sum(traj_predict_single==predict_b1)
sum(traj_predict_single>0)
sum(traj_predict_single>=0)
traj_predict_convert = result_convertClassCell(traj_predict, model_data.convert2database(:, 2));
prediction{i}(indic)
convertTable(prediction{i}(indic))
%-- 2013-07-11 08:37 PM --%
load('evaluate.mat', 'traj_predict', 'traj_scores')
addpath('./fish_recog');
model_data = load('data_23Species.mat');
traj_predict_convert = result_convertClassCell(traj_predict, model_data.convert2database(:, 2));
traj_predict_convert(:,2)=traj_predict;
clear;
sample_predictSpecies
load('10.evaluate.mat', 'traj_predict')
load('10.evaluate.mat', 'traj_scores')
clear;
load('1.fish27k_attributes.mat', 'traj_id')
load('10.evaluate.mat', 'GMM_result_score')
load('10.evaluate.mat', 'BGOT_result_scores_23')
load('7.result_fromUnormalized_20130708.mat', 'trajs_classid_vote')
[traj_predict, traj_scores] = result_trajvote_byScore( GMM_result_score, traj_id);
load('10.evaluate.mat', 'GMM_result_predict')
CWI_scores = GMM_result_score(GMM_result_predict>0);
CWI_traj = traj_id(GMM_result_predict>0);
sum(CWI_scores)
CWI_scores = GMM_result_score(GMM_result_predict>0,:);
sum(CWI_scores)
find(sum(CWI_scores,2)==0)
find(sum(CWI_scores,2)>=0)
[mv,CWI_predict] = max(CWI_scores);
[mv,CWI_predict] = max(CWI_scores,[],2);
unique(CWI_traj);
CWI_traj_id = unique(CWI_traj);
for i = 1:8087
traj_indic = CWI_traj == CWI_traj_id(i);
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i}=unique(CWI_predict(traj_indic));
CWI_traj_scores{i}=tmp_score(:,CWI_traj_predict{i});
end
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}tmp_score(:,CWI_traj_predict{i});
end
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
trajs_unique = unique(traj_id);
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==traj_id);
end
CWI_traj_classid_vote = traj_classid_vote(indic);
for i = 1:8087
indic1(i,1)=sum(ismember(CWI_traj_classid_vote(i), CWI_traj_predict{i}));
end
for i = 1:8663
indic2(i,1)=sum(ismember(traj_classid_vote(i), traj_predict{i}));
end
sum(indic1)
sum(indic2)
find(indic2==0)
tmp1 = traj_classid_vote(find(indic2==0));
tmp1(:,2) = traj_predict(find(indic2==0));
tmp2 = traj_predict(find(indic2==0));
find(indic2==0)
tmp1 = traj_classid_vote(find(indic2==0));
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==traj_id);
end
tmp = indic2(indic);
find(tmp==0&indic1>0)
find(tmp>0&indic1==0)
find(tmp==0&indic1>0)
CWI_traj_scores(8011)
CWI_traj_id(8011)
load('10.evaluate.mat', 'traj_scores')
load('10.evaluate.mat', 'GMM_result_score')
load('1.fish27k_attributes.mat', 'traj_id')
GMM_result_score(traj_id == CWI_traj_id(8011),:)
median(GMM_result_score(traj_id == CWI_traj_id(8011),:))
mean(GMM_result_score(traj_id == CWI_traj_id(8011),:))
for i = 1:23
recall(i,1)=sum(indic1(CWI_traj_classid_vote==i))/sum(CWI_traj_classid_vote==i);
recall(i,2)=sum(indic2(CWI_traj_classid_vote==i))/sum(CWI_traj_classid_vote==i);
for i = 1:23
recall(i,1)=sum(indic1(CWI_traj_classid_vote==i))/sum(CWI_traj_classid_vote==i);
recall(i,2)=sum(indic2(traj_classid_vote==i))/sum(traj_classid_vote==i);
end
load('9_20130711_GMM_result_cv5.mat', 'traj_scores_b1')
load('9_20130711_GMM_result_cv5.mat', 'scores_b')
[ traj_predict, traj_scores] = result_trajvote_byScore( scores_b, traj_id );
CWI_traj = traj_id(sum(scores_b)>0);
CWI_traj = traj_id(sum(scores_b,[],2)>0);
CWI_traj = traj_id(sum(scores_b,2)>0);
CWI_scores = scores_b(sum(scores_b,2)>0);
CWI_scores = scores_b(sum(scores_b,2)>0,:);
CWI_traj_id = unique(CWI_traj);
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
[mv,CWI_predict] = max(CWI_scores,[],2);
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
trajs_unique = unique(traj_id);
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==traj_id);
end
CWI_traj_classid_vote = traj_classid_vote(indic);
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==trajs_unique(i));
end
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==trajs_unique);
end
CWI_traj_classid_vote = traj_classid_vote(indic);
CWI_traj_classid_vote = traj_classid_vote(CWI_indic);
for i = 1:6585
indic1(i,1)=sum(ismember(CWI_traj_classid_vote(i), CWI_traj_predict{i}));
end
for i = 1:8663
indic2(i,1)=sum(ismember(traj_classid_vote(i), traj_predict{i}));
end
sum(indic1)
sum(indic2)
load('9_20130711_BGOT_result_cv5.mat', 'BGOT_result_scores')
load('9_20130711_evaluate_result.mat', 'BGOT_result_scores_whole')
[ traj_predict, traj_scores] = result_trajvote_byScore( BGOT_result_scores_whole, traj_id );
CWI_traj = traj_id(sum(BGOT_result_scores_whole,2)>0);
CWI_scores = BGOT_result_scores_whole(sum(scores_b,2)>0,:);
CWI_traj_id = unique(CWI_traj);
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
[mv,CWI_predict] = max(CWI_scores,[],2);
[ traj_predict, traj_scores] = result_trajvote_byScore( BGOT_result_scores_whole, traj_id );
CWI_traj = traj_id(sum(BGOT_result_scores_whole,2)>0);
CWI_scores = BGOT_result_scores_whole(sum(scores_b,2)>0,:);
CWI_traj_id = unique(CWI_traj);
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
[mv,CWI_predict] = max(CWI_scores,[],2);
CWI_scores = BGOT_result_scores_whole(sum(BGOT_result_scores_whole,2)>0,:);
CWI_traj_id = unique(CWI_traj);
[mv,CWI_predict] = max(CWI_scores,[],2);
CWI_traj_id = unique(CWI_traj);
for i = 1:8087
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
for i = 1:8663
traj_indic = find(CWI_traj == CWI_traj_id(i));
tmp_score = CWI_scores(traj_indic,:);
CWI_traj_predict{i,1}=unique(CWI_predict(traj_indic));
if length(traj_indic) > 1
CWI_traj_scores{i,1}=median(tmp_score(:,CWI_traj_predict{i}));
else
CWI_traj_scores{i,1}=tmp_score(:,CWI_traj_predict{i});
end
end
trajs_unique = unique(traj_id);
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==traj_id);
end
CWI_traj_classid_vote = traj_classid_vote(indic);
for i = 1:8663
indic(i,1)=find(CWI_traj_id(i)==traj_id);
end
CWI_traj_classid_vote = traj_classid_vote(indic);
for i = 1:8087
indic(i,1)=find(CWI_traj_id(i)==trajs_unique);
end
for i = 1:8663
indic(i,1)=find(CWI_traj_id(i)==trajs_unique);
end
CWI_traj_classid_vote = traj_classid_vote(CWI_indic);
for i = 1:6585
indic1(i,1)=sum(ismember(CWI_traj_classid_vote(i), CWI_traj_predict{i}));
end
for i = 1:8663
indic2(i,1)=sum(ismember(traj_classid_vote(i), traj_predict{i}));
end
for i = 1:8663
indic1(i,1)=sum(ismember(CWI_traj_classid_vote(i), CWI_traj_predict{i}));
end
for i = 1:8663
indic2(i,1)=sum(ismember(traj_classid_vote(i), traj_predict{i}));
end
sum(indic1)
sum(indic2)
traj_predict(:,2)=CWI_traj_predict;
clear;
sample_predictSpecies
features = zeros(10,2626);
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[BGOT_result_predict, BGOT_result_scores_23] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
if iscell(BGOT_result_scores_23)
BGOT_result_scores_23 = cell2mat(BGOT_result_scores_23);
end
BGOT_result_score = zeros(length(BGOT_result_predict), 1);
for i = 1:length(BGOT_result_predict)
BGOT_result_score(i,1) = BGOT_result_scores_23(i, BGOT_result_predict(i));
end
[GMM_result_predict, GMM_result_score,  GMM_result_score_all] = interface_rejection( BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23, features, model_data );
features = zeros(10,2626);
[BGOT_result_predict, BGOT_result_scores_23] = classify_HierSVM_predict(features, model_data.BGOT_1301_69f);
if iscell(BGOT_result_scores_23)
BGOT_result_scores_23 = cell2mat(BGOT_result_scores_23);
end
BGOT_result_score = zeros(length(BGOT_result_predict), 1);
for i = 1:length(BGOT_result_predict)
BGOT_result_score(i,1) = BGOT_result_scores_23(i, BGOT_result_predict(i));
end
[GMM_result_predict, GMM_result_score,  GMM_result_score_all] = interface_rejection( BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23, features, model_data );
clear;
load('2.fish90k_fishImage.mat')
load('1.fish90k_attributes_20121121.mat', 'class_id')
maskImg = maskImg(class_id>0);
help polyroi
help roipoly
help polyroi
help roipoly
load('0.FishDatabaseGT_reviewed_20130227.mat', 'GTInfo')
load('1.fish90k_attributes_20121121.mat', 'class_id')
GTInfo = GTInfo(class_id>0);
GTInfos = GTInfo.ID(class_id>0);
for i = 1:27470
contours = GTInfos(i).database_info.contour;
end
for i = 1:27470
contours{i,1} = GTInfos(i).database_info.contour;
end
for i = 1:27470
contours{i,1} = GTInfos(i).database_info.contour;
end
for i = 1:27470
if isfield(GTInfos(i).database_info, 'contour')
contours{i,1} = GTInfos(i).database_info.contour;
end
end
for i = 1:27470
if isfield(GTInfos(i).database_info, 'contour')
contour = GTInfos(i).database_info.contour;
contours{i,1}={1:2:end};
contours{i,2}={2:2:end};
end
end
for i = 1:27470
if isfield(GTInfos(i).database_info, 'contour')
contour = GTInfos(i).database_info.contour;
contours{i,1}=contour(1:2:end);
contours{i,2}=contour(2:2:end);
end
end
for i = 1:27470
if isfield(GTInfos(i).database_info, 'contour')
contour = GTInfos(i).database_info.contour;
contours{i,1}=contour(1:2:end);
contours{i,2}=contour(2:2:end);
box(i,1)=min(contours{i,1});box(i,2)=max(contours{i,1});
box(i,3)=min(contours{i,2});box(i,4)=max(contours{i,2});
end
end
boxs = box(sum(box)>0);
boxs = box(sum(box,[],2)>0);
boxs = box(sum(box,2)>0);
boxs = box(sum(box,2)>0,:);
sizes(:,1)=boxs(:,2)-boxs(:,1);
sizes(:,2)=boxs(:,4)-boxs(:,3);
plot(histc(sizes(:,1),1:200))
figure(2);
plot(histc(sizes(:,2),1:200))
help regionprops
load('2.fish90k_fishImage.mat')
maskImg = maskImg(class_id>0);
maskImg2 = append_cleanBinaryImage(maskImg{1});
stats = regionprops(binImg_rot,'basic');
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box = stats.BoundingBox(maxindex);
bounding_box = stats(maxindex).BoundingBox;
for i = 1:27470
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(binImg_rot,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box{i,1} = stats.BoundingBox(maxindex);
end
for i = 1:27470
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box{i,1} = stats.BoundingBox(maxindex);
end
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
for i = 1:27470
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box{i,1} = stats(maxindex).BoundingBox;
end
for i = 1:27470
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box{i,1} = stats(maxindex).BoundingBox;
end
for i = 24041:27470
maskImg2 = append_cleanBinaryImage(maskImg{i});
stats = regionprops(maskImg2,'basic');
[maxarea, maxindex] = max([stats.Area]);
bounding_box{i,1} = stats(maxindex).BoundingBox;
end
for i = 1:27470
bounding_boxs(i,:) = bounding_box(i);
end
for i = 1:27470
bounding_boxs(i,:) = bounding_box{i};
end
help regionprops
min(bounding_boxs(:,3))
max(bounding_boxs(:,3))
max(bounding_boxs(:,4))
min(bounding_boxs(:,4))
image_accpet = ones(length(maskImg), 1);
[image_accpet] = append_acceptImage(maskImg);
tabulate(image_accpet)
clear;
sample_predictSpecies
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'traj_id', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[ BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23 ] = interface_classification( features, model_data.BGOT_1301_69f );
[GMM_result_predict, GMM_result_score,  GMM_result_score_23] = interface_rejection( BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23, features, model_data );
GMM_result_predict(GMM_result_predict>0) = model_data.convert2database(GMM_result_predict(GMM_result_predict>0), 2);
[traj_predict, traj_scores] = result_trajvote_byScore( GMM_result_score_23, traj_id);
traj_predict_convert = result_convertClassCell(traj_predict, model_data.convert2database(:, 2));
%-- 2013-07-12 12:16 AM --%
sample_predictSpecies
load('0.fish7k2_image.mat', 'fish_rgbimg', 'fish_mskimg')
load('0.fish7k2_attribute.mat', 'class_id')
load('0.fish7k2_attribute.mat', 'traj_id')
[ detection_predict, detection_score, traj_predict, traj_scores ] = interface_recognizeFishFromImage(model_data , fish_rgbimg, fish_mskimg,  traj_id);
%-- 2013-07-20 12:40 AM --%
%-- 2013-07-25 08:04 PM --%
load('4.detection_random.mat')
[ fish_image, fish_mask, fish_whole, fish_id, traj_id ] = draw_detections( video_id, detection, videos );
clear;
load('2.fish90k_feature_121030.mat')
load('0.FishDatabaseGT_reviewed_20130227.mat')
load('1.fish90k_attributes_20121121.mat', 'class_id')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
sum(features(1,:)==features_90k(1,:))
load('1.fish90k_attributes_20121121.mat', 'traj_id')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
load('2.fish90k_feature_23species_130407_prefixNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureSet(features)
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
mean(features);
sum( mean(features)  == model_data.feature_mean)
mean(features);
[features0, prefixValue]= interface_normalizeFeatureSet(features);
[features0]= interface_normalizeFeatureSet(features,prefixValue);
[features1]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
unique_traj = unique(traj_id);
class_id_b = class_id;
class_id_b(class_id>0)=1;
tabulate(class_id_b)
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, 10, 2, 0);
[  result] = classify_crossValidation(features, featurei, class_id_b, traj_id, 10, 2, 0);
load('matlab.mat', 'traj_id', 'traj_indices')
for i = 1:10
cv_id(ismember(traj_id, traj_id_unique(traj_indices==i)))=i;
end
cv_id = cv_id';
tabulate(cv_id)
unique(traj_id(cv_id==1));
ans2=traj_id_unique(traj_indices==1);
for i = 1:10
[model(i)] = classify_SVM_train(features(cv_id~=i), class_id_b(cv_id~=i), 2);
end
for i = 1:10
[model(i)] = classify_SVM_train(features(cv_id~=i), class_id_b(cv_id~=i), 2);
end
for i = 1:5
i
[model(i)] = classify_SVM_train(features(cv_id~=i), class_id_b(cv_id~=i), 2);
end
%-- 2013-07-25 10:19 PM --%
load('2.fish90k_feature_121030_unnormalize.mat', 'features')
load('1.attribute.mat', 'class_id_b', 'cv_id')
for i = 6:10
i
[model(i)] = classify_SVM_train(features(cv_id~=i), class_id_b(cv_id~=i), 2);
end
%-- 2013-07-25 10:50 PM --%
load('data_featureSelection_Species_0008.mat')
load('data_featureSelection_Species_0001.mat', 'scoreResult')
for i = 1:23
load('data_featureSelection_Species_0001.mat')
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv,mi]=scoreResult(:,3);
subSet{i,1}=tmp_data.featureSubset(1:mi);
end
for i = 1:23
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv,mi]=tmp_data.scoreResult(:,3);
subSet{i,1}=tmp_data.featureSubset(1:mi);
end
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv,mi]=tmp_data.scoreResult(:,3);
for i = 1:23
tmp_data = load(sprintf('data_featureSelection_Species_%04d.mat',i));
[mv,mi]=max(tmp_data.scoreResult(:,3));
subSet{i,1}=tmp_data.featureSubset(1:mi);
end
load('9.result_20130529_species1_15_BGOT_data.mat', 'featureSub_20130415Revise')
sort(featureSub_20130415Revise{1})
feature_subSet{1}=[feature_subSet{1},featureSub_20130415Revise{1}]
sort(featureSub_20130415Revise{2})
feature_subSet{2}=[feature_subSet{2},featureSub_20130415Revise{2}];
feature_subSet{2}=[50,152];
sort(featureSub_20130415Revise{4})
feature_subSet{2}=[feature_subSet{2},featureSub_20130415Revise{4}];
feature_subSet{6}=featureSub_20130415Revise{10};
feature_subSet{8}=featureSub_20130415Revise{7};
feature_subSet{11}=featureSub_20130415Revise{12};
feature_subSet{14}=featureSub_20130415Revise{9};
feature_subSet{18}=featureSub_20130415Revise{11};
feature_subSet{16}=[feature_subSet{16},featureSub_20130415Revise{6}];
feature_subSet_Revised1_from15species{16}=[feature_subSet_Revised1_from15species{16},featureSub_20130415Revise{6}];
load('data_23Species.mat', 'BGOT_1301_69f')
rootSub = BGOT_1301_69f.Root.Subfeature;
load('data_23Species.mat', 'feature_index')
tmp_sub = find(ismember(feature_index, rootSub));
tmp_index = randperm(623);
feature_subSet_Revised2_from15species = feature_subSet_Revised1_from15species;
feature_subSet_Revised1_from15species([9,15,20,22])=tmp_sub(tmp_index(1:40));
feature_subSet_Revised1_from15species{[9,15,20,22]}=tmp_sub(tmp_index(1:40));
feature_subSet_Revised1_from15species([9,15,20,22])={tmp_sub(tmp_index(1:40))};
clear;
load('1.attributes.mat', 'feature_subSet_Revised2_fromRoosub')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'class_id')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('9.result_20130530_NIPS_result.mat', 'train_index_train4')
load('9.result_20130530_NIPS_result.mat', 'predict')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
for i = 1:5
i
indic_train = train_index_train4(:,i)>0;
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), 1:2626, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(~indic_train,:), hier_arch(i) );
end
%-- 2013-07-25 11:24 PM --%
load('matlab.mat')
for i = 1:5
i
indic_train = train_index_train4(:,i)>0;
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( feature_train(indic_train,:), classid_train(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_index_train4(:,i)>0;
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_index_train4(:,i)>0;
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
load('1.attributes.mat', 'feature_subSet_Revised2_fromRoosub')
load('1.attributes.mat', 'feature_subSet_Revised1_from15species')
load('data_23Species.mat', 'BGOT_1301_69f')
rootSub = BGOT_1301_69f.Root.Subfeature;
load('data_23Species.mat', 'feature_index')
tmp_sub = find(ismember(feature_index, rootSub));
tmp_index = randperm(623);
feature_subSet_Revised2_from15species = feature_subSet_Revised1_from15species;
feature_subSet_Revised1_from15species([9,15,20,22])=tmp_sub(tmp_index(1:40));
feature_subSet_Revised1_from15species{[9,15,20,22]}=tmp_sub(tmp_index(1:40));
feature_subSet_Revised1_from15species([9,15,20,22])={tmp_sub(tmp_index(1:40))};
for i = 2:5
i
indic_train = train_index_train4(:,i)>0;
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 2:5
i
indic_train = train_index_train4(:,i)>0;
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
load('1.attributes.mat', 'feature_subSet')
load('1.attributes.mat', 'feature_subSet_Revised1_from15species')
load('1.attributes.mat', 'feature_subSet_Revised2_fromRoosub')
clear;
load('1.attributes.mat', 'feature_subSet_Revised2_fromRoosub')
load('matlab.mat', 'class_id', 'features', 'train_id', 'test_id')
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSub{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
load('1.attributes.mat', 'feature_subSet_Revised2_fromRoosub')
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 4:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:2
for j = 1:23
scores{i}(j,:)=classify_GMM_predict(features(test_id{i}, models{i}{j});
for i = 1:2
for j = 1:23
scores{i}(j,:)=classify_GMM_predict(features(test_id{i}), models{i}{j});
end
end
for i = 1:2
for j = 1:23
scores{i}(j,:)=classify_GMM_predict(features(test_id{i},:), models{i}{j});
end
end
for i = 1:2
for j = 1:23
if ~isempty(models{i}{j}.indic)
scores{i}(j,:)=classify_GMM_predict(features(test_id{i},:), models{i}{j});
end
end
end
for i = 1:2
for j = 1:23
if ~isempty(models{i}{j}.indic)
tmp_score=classify_GMM_predict(features(test_id{i},:), models{i}{j});
scores{i}(:,j)=log(eps+tmp_score)-log(eps);
end
end
end
for i = 3:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 3:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
%-- 2013-07-27 08:48 PM --%
load('matlab.mat')
hier_tree_1301_69a.Node_Rejection = 0;
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'featurei')
for i = 1:5
i
indic_train = train_id{i};
indic_test = test_id{i};
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), featurei, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(indic_test,:), hier_arch(i) );
end
for i = 1:5
result_predict_all(test_id{i})=result_predict{i};
result_score_all(test_id{i})=result_score{i};
result_scores_23_all(test_id{i})=result_scores_23{i};
end
for i = 1:5
result_predict_all(test_id{i},:)=result_predict{i};
result_score_all(test_id{i},:)=result_score{i};
result_scores_23_all(test_id{i},:)=result_scores_23{i};
end
for i = 1:5
result_predict_all(test_id{i},:)=result_predict{i};
result_score_all(test_id{i},:)=result_score{i};
result_scores_23_all(test_id{i},:)=result_scores_23{i};
end
for i = 1:5
result_predict_all(test_id{i},:)=result_predict{i};
result_score_all(test_id{i},:)=result_score{i};
result_scores_23_all(test_id{i},:)=result_scores_23{i};
end
result_evaluate(result_predict_all, class_id)
load('2.BGOT_result_NodeRejct1.mat', 'result_predict_all')
sum(result_predict_all==result_predict_all1)
%-- 2013-07-29 02:57 PM --%
load('2.fish90k_feature_23species_130407_prefixNormalize95.mat', 'features', 'class_id', 'featurei')
load('1.attributes.mat', 'train_id', 'test_id')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
hier_tree_1301_69a = 1;
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
hier_tree_1301_69a.Node_Rejection = 1;
for i = 1:5
i
indic_train = train_id{i};
indic_test = test_id{i};
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), featurei, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(indic_test,:), hier_arch(i) );
end
load('1.attributes.mat', 'feature_subSet_Revised3')
[ models, featureSubNew, componentNum ] = rejection_applyFS( features, class_id, feature_subSet_Revised3 );
load('3.GMM_result.mat', 'models')
models = models{1};
train_id = train_id{1};
test_id = find(~ismember(1:27370, train_id));
test_id = find(~ismember(1:27370, train_id))';
for j = 1:23
if ~isempty(models{j}.indic)
tmp_score=classify_GMM_predict(features(test_id,:), models{j});
scores_train(:,j)=log(eps+tmp_score)-log(eps);
end
end
e=classify_GMM_predict(features(test_id,:), models{j});
scores_train(:,j)=log(eps+tmp_score)-log(eps);
end
end
for j = 1:23
if ~isempty(models{j}.indic)
tmp_score=classify_GMM_predict(features(test_id,:), models{j});
scores_test(:,j)=log(eps+tmp_score)-log(eps);
tmp_score=classify_GMM_predict(features(train_id,:), models{j});
scores_train(:,j)=log(eps+tmp_score)-log(eps);
end
end
for j = 1:23
if ~isempty(models{j}.indic)
tmp_score=classify_GMM_predict(features(test_id,:), models{j});
scores_test(:,j)=log(eps+tmp_score)-log(eps);
tmp_score=classify_GMM_predict(features(train_id,:), models{j});
scores_train(:,j)=log(eps+tmp_score)-log(eps);
end
end
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
for j = 1:23
if ~isempty(models{j}.indic)
tmp_score=classify_GMM_predict(features(test_id,:), models{j});
scores_test(:,j)=log(eps+tmp_score)-log(eps);
tmp_score=classify_GMM_predict(features(train_id,:), models{j});
scores_train(:,j)=log(eps+tmp_score)-log(eps);
end
end
sum(models{1,1}.indic)
classify_GMM_predict(features(test_id,:), models{1});
tmp_score=classify_GMM_predict(features(train_id,:), models{1});
load('4.GMM_scores.mat', 'scores')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
load('3.GMM_result.mat', 'models')
sum(models{1,5}{23,1}.indic)
=classify_GMM_predict(features(1,:), models{1}{1});
classify_GMM_predict(features(1,:), models{1}{1});
classify_GMM_predict(features(1,:), models{2}{1});
classify_GMM_predict(features(1,:), models{3}{1});
classify_GMM_predict(features(1,:), models{4}{1});
%-- 2013-07-29 03:39 PM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('3.GMM_result.mat', 'models')
model_data = load('data_23Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
=classify_GMM_predict(features(1,:), models{1}{1});
classify_GMM_predict(features(1,:), models{1}{1});
classify_GMM_predict(features(1,:), models{1}{1})
classify_GMM_predict(features(1,:), models{2}{1})
classify_GMM_predict(features(1,:), models{3}{1})
classify_GMM_predict(features(1,:), models{4}{1})
classify_GMM_predict(features(1,:), models{5}{1})
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
classify_GMM_predict(features(1,:), models{1}{1})
classify_GMM_predict(features(1,:), models{2}{1})
classify_GMM_predict(features(1,:), models{3}{1})
classify_GMM_predict(features(1,:), models{4}{1})
classify_GMM_predict(features(1,:), models{5}{1})
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('2.fish90k_feature_23species_130407_prefixNormalize95.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
predict1 = interface_classification( features_normalized, model_data.BGOT_1301_69f );
predict2 = interface_classification( features_prefixNormalize, model_data.BGOT_1301_69f );
predict3 = interface_classification( features_prefixNormalize95, model_data.BGOT_1301_69f );
predict1 = interface_classification( features_normalized, model_data.BGOT_1301_69f );
predict2 = interface_classification( features_prefixNormalize, model_data.BGOT_1301_69f );
predict3 = interface_classification( features_prefixNormalize95, model_data.BGOT_1301_69f );
predict1 = interface_classification( features_normalized, model_data.BGOT_1301_69f );
result_evaluate(predict1, class_id)
%-- 2013-07-29 04:06 PM --%
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'features')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'class_id')
model_data = load('data_23Species.mat');
predict2 = interface_classification( features_normalized, model_data.BGOT_1301_69f );
result_evaluate(predict2, class_id)
predict2 = interface_classification( features, model_data.BGOT_1301_69f );
result_evaluate(predict2, class_id)
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('1.attributes.mat', 'train_id', 'test_id', 'feature_subSet_Revised3')
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
model_data = load('data_23Species.mat');
for i = 3:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
for j = 1:23
if ~isempty(models{i}{j}.indic)
tmp_score=classify_GMM_predict(features(test_id{i},:), models{i}{j});
scores{i}(:,j)=log(eps+tmp_score)-log(eps);
end
end
end
for i = 1:5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 1:5
for j = 1:23
if ~isempty(models{i}{j}.indic)
tmp_score=classify_GMM_predict(features(test_id{i},:), models{i}{j});
scores{i}(:,j)=log(eps+tmp_score)-log(eps);
end
end
end
for i = 5
for j = 1:23
if ~isempty(models{i}{j}.indic)
tmp_score=classify_GMM_predict(features(test_id{i},:), models{i}{j});
scores{i}(:,j)=log(eps+tmp_score)-log(eps);
end
end
end
for i = 5
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
for i = 4
i
indic_train = train_id{i};
[ models{i}, featureSubNew{i}, componentNum{i} ] = rejection_applyFS( features(indic_train,:), class_id(indic_train), featureSub );
end
%-- 2013-07-29 07:31 PM --%
load('video1_cut_xml.mat', 'video_1_cut1_detects_num', 'video_1_cut2_detects_num')
[ fish_image, mask_image, draw_image ] = recognizeFish( 'gt_Video1_cut_1.avi', video_1_cut1_detects_num );
imshow(draw_image{2})
for i = 1:6000
imshow(draw_image{i});
pause();
end
for i = 1:6239
imwrite(draw_image{i},sprintf('%04d.png',i));
for i = 1:6239
imwrite(draw_image{i},sprintf('tmp/%04d.png',i));
end
[ fish_image, mask_image, draw_image ] = recognizeFish( 'gt_Video1_cut_1.avi', video_1_cut1_detects_num );
for i = 1:length(draw_image)
imwrite(draw_image{i},sprintf('tmp/%04d.png',i));
end
[ fish_image, mask_image, draw_image ] = recognizeFish( 'gt_Video1_cut_1.avi', video_1_cut1_detects_num );
for i = 1:length(draw_image)
imwrite(draw_image{i},sprintf('tmp/%04d.png',i));
end
[ fish_image, mask_image, draw_image ] = recognizeFish( 'gt_Video1_cut_1.avi', video_1_cut1_detects_num );
for i = 1:length(draw_image)
imwrite(draw_image{i},sprintf('tmp/%04d.png',i));
end
[ fish_image, mask_image, draw_image ] = recognizeFish( 'gt_Video1_cut_1.avi', video_1_cut1_detects_num );
for i = 1:length(draw_image)
imwrite(draw_image{i},sprintf('tmp/%04d.png',i));
end
[ fish_image_cut2, mask_image_cut2, draw_image_cut2 ] = recognizeFish( 'gt_Video1_cut_2.avi', video_1_cut1_detects_num );
for i = 1:length(draw_image)
imwrite(draw_image{i},sprintf('cut2/%04d.png',i));
end
for i = 1:length(draw_image)
imwrite(draw_image_cut2{i},sprintf('cut2/%04d.png',i));
end
for i = 1:length(draw_image_cut2)
imwrite(draw_image_cut2{i},sprintf('cut2/%04d.png',i));
end
[ fish_image_cut2, mask_image_cut2, draw_image_cut2 ] = recognizeFish( 'gt_Video1_cut_2.avi', video_1_cut2_detects_num );
for i = 1:length(draw_image_cut2)
imwrite(draw_image_cut2{i},sprintf('cut2/%04d.png',i));
end
load('video2_cut_xml.mat', 'detects_xml_cut1_num', 'detects_xml_cut2_num')
[ fish_image_cut1, mask_image_cut1, draw_image_cut1 ] = recognizeFish( 'gt_Video2_cut_1.avi', detects_xml_cut1_num );
for i = 1:length(draw_image_cut1)
imwrite(draw_image_cut1{i},sprintf('cut1/%04d.png',i));
end
[ fish_image_cut2, mask_image_cut2, draw_image_cut2 ] = recognizeFish( 'gt_Video2_cut_2.avi', detects_xml_cut2_num );
for i = 1:length(draw_image_cut2)
imwrite(draw_image_cut2{i},sprintf('cut2/%04d.png',i));
end
clear;
cls;
load('images.mat', 'fish_image_cut1', 'mask_image_cut1')
%-- 2013-07-29 10:45 PM --%
load('images.mat', 'fish_image_cut2', 'mask_image_cut2')
load('video1_cut_xml.mat', 'traj_id_cut2')
model_data = load('data_23Species.mat');
[ detection_predict_video1_cut2, detection_score_video1_cut2, traj_predict_video1_cut2, traj_scores_video1_cut2, traj_array_video1_cut2 ] = interface_recognizeFishFromImage( model_data, fish_image_cut2, mask_image_cut2, traj_id_cut2 );
features_video1_cut2 = features_raw;
tabulate(detection_predict_video1_cut2)
[ detection_predict_video1_cut2, detection_score_video1_cut2, traj_predict_video1_cut2, traj_scores_video1_cut2, traj_array_video1_cut2 ] = interface_recognizeFishFromImage( model_data, fish_image_cut2, mask_image_cut2, traj_id_cut2 );
load('features_unormalized.mat', 'features_video1_cut2')
[ detection_predict_video1_cut2, detection_score_video1_cut2, traj_predict_video1_cut2, traj_scores_video1_cut2, traj_array_video1_cut2 ] = interface_recognizeFishFromImage( model_data, fish_image_cut2, mask_image_cut2, traj_id_cut2 );
load('features_unormalized.mat', 'features_video1_cut2')
tabulate(detection_predict_video1_cut2)
model_data = load('data_23Species.mat');
[ detection_predict_video1_cut2, detection_score_video1_cut2, traj_predict_video1_cut2, traj_scores_video1_cut2, traj_array_video1_cut2 ] = interface_recognizeFishFromImage( model_data, fish_image_cut2, mask_image_cut2, traj_id_cut2 );
load('features_unormalized.mat', 'features_video1_cut2')
tabulate(detection_predict_video1_cut2)
%-- 2013-07-30 04:10 PM --%
load('2.fish90k_feature_23species_130226_selfNormalize95.mat', 'class_id', 'traj_id', 'featurei', 'cv_id_fold5', 'features')
load('4.fish27k_hierTree_1301_20130430.mat', 'hier_tree_1301_69a')
load('1.attributes.mat', 'train_id', 'test_id')
for i = 1:5
i
indic_train = train_id{i};
indic_test = test_id{i};
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), featurei, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(indic_test,:), hier_arch(i) );
end
for i = 1:5
resut_predict_all(test_id{i}) = result_predict{i};
end
tabulate(resut_predict_all)
result_evaluate(resut_predict_all', class_id)
find(train_id==1)
find(train_id{1}==1)
find(train_id{2}==1)
find(train_id{3}==1)
find(train_id{4}==1)
find(train_id{5}==1)
find(train_id{3}==53)
test_id2{1}=test_id{2};
test_id2{2}=test_id{3};
test_id2{3}=test_id{4};
test_id2{4}=test_id{5};
test_id2{5}=test_id{1};
load('2.speces_23.mat', 'result_predict')
sum(result_predict==result_predict_all)
sum(result_predict==resut_predict_all)
sum(result_predict==resut_predict_all')
for i = 1:5
i
indic_train = train_id{i};
indic_test = test_id2{i};
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), featurei, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(indic_test,:), hier_arch(i) );
end
for i = 1:5
resut_predict_all(test_id{i}) = result_predict{i};
end
for i = 1:5
resut_predict_all(test_id2{i}) = result_predict{i};
end
resut_predict_all = resut_predict_all';
tabulate(resut_predict_all)
result_evaluate(resut_predict_all, class_id)
load('1.attributes.mat', 'test_id')
test_id2{1}=test_id{5};
test_id2{2}=test_id{1};
test_id2{3}=test_id{2};
test_id2{4}=test_id{3};
test_id2{5}=test_id{4};
for i = 1:5
i
indic_train = train_id{i};
indic_test = test_id2{i};
[hier_arch(i)] = classify_HierSVM_train(features(indic_train,:), class_id(indic_train), featurei, hier_tree_1301_69a, 2);
[ result_predict{i}, result_score{i}, result_scores_23{i} ] = interface_classification( features(indic_test,:), hier_arch(i) );
end
for i = 1:5
resut_predict_all(test_id2{i},1) = result_predict{i};
end
tabulate(resut_predict_all)
result_evaluate(resut_predict_all, class_id)
clear;
load('2.speces_23_cv5_convert.mat', 'scores')
clear;
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'featurei')
addpath('./fish_recog');
model_data = load('data_15Species.mat');
[features]= interface_normalizeFeatureFromPrefix(features, model_data.feature_mean, model_data.feature_std);
[result_predict, result_scores] = classify_HierSVM_predict(features, model_data.BGOT_Trained);
if iscell(result_scores)
result_scores = cell2mat(result_scores);
end
[ result_predict ] = append_convertlabel( model_data.convert2database, result_predict);
clear;
load('1.video_id.mat', 'video_id')
videolist.video_id = video_id;
script_downloadVideo_fromGTInfo
%-- 2013-08-15 09:28 PM --%
load('1.video_id.mat', 'video_id')
[ fish_detection, url ] = interface_downloadVideo( video_id(1) );
video_id
[ fish_detection, url ] = interface_downloadVideo( video_id(1) );
for i = 1:70
[ fish_detection(i,1), url(i,1) ] = interface_downloadVideo( video_id(1) );
end
for i = 10:70
[ fish_detection(i,1), url(i,1) ] = interface_downloadVideo( video_id(1) );
end
for i = 1:70
[ fish_detection(i,1), url(i,1) ] = interface_downloadVideo( video_id(1) );
end
[ fish_detection, url ] = interface_downloadVideo( video_id );
for i = 1:70
lens(i,1)=lengt(fish_detection{1,1}.fish_id);
end
for i = 1:70
lens(i,1)=length(fish_detection{i}.fish_id);
end
for i = 1:70
[ fish_detection(i,1), url(i,1) ] = interface_downloadVideo( video_id(i) );
i
length(fish_detection{i}.fish_id)
end
for i = 1:15
[ fish_detection(i,1), url(i,1) ] = interface_downloadVideo( video_id(i) );
i
length(fish_detection{i}.fish_id)
end
load('2.detections.mat')
fish_detection(1:13)=fish_detection1;
%-- 2013-08-15 11:30 PM --%
%-- 2013-08-15 11:42 PM --%
load('2.detections.mat', 'fish_detection', 'url')
load('1.video_id.mat', 'video_id')
[ recog_images, recog_result ] = interface_recognizeFish( video_id{54}, fish_detection{54} )
output_mov_buffer_high_res
[ recog_images, recog_result ] = interface_recognizeFish( video_id{54}, fish_detection{54} )
frames = frames(1:50);
[ recog_images, recog_result ] = interface_recognizeFish( video_id{54}, fish_detection{54} )
[ recog_images, recog_result ] = interface_recognizeFish( video_id{2}, fish_detection{2} )
imshow(recog_images{1})
imshow(recog_images.fish_image{1})
imshow(recog_images.mask_image{1})
load('2.detections.mat', 'fish_detection')
[ recog_images, recog_result ] = interface_recognizeFish( video_id{4}, fish_detection{4} )
isempty(fish_detection);
isempty(fish_detection.fish_id);
[ recog_images, recog_result ] = interface_recognizeFish( video_id{4}, fish_detection{4} )
for i = 1
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat]
for i = [2,4]
i
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = [4,2]
i
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
(['tmp/' video_id{i} '.mat']
['tmp/' video_id{i} '.mat']
save(['tmp/' video_id{i} '.mat'], 'ans');
for i = [4,2]
i
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
%-- 2013-08-16 01:01 AM --%
load('1.video_id.mat')
for i = 16:30
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
for i = 16:30
i
[ fish_detection(i,1) ] = interface_downloadVideo( video_id(i) );
[ recog_images, recog_result ] = interface_recognizeFish( video_id{i}, fish_detection{i} );
save(['tmp/' video_id{i} '.mat'], 'recog_images', 'recog_result');
end
%-- 2013-08-16 02:26 PM --%
mov_properties = mmreader('0002b5824b5d5c46f818828759c97420#201210021120.flv');
imshow(read(mov_properties, 287));
load('2.detections.mat', 'fish_detection')
unique_frames = unique(fish_detection{56,1}.frame_id);
for i = 1:60
read(mov_properties, unique_frames(i)+1);
end
imshow(read(mov_properties, 287));
%-- 2013-08-16 02:30 PM --%
load('2.detections.mat')
load('1.video_id.mat')
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
unique_frames(k)+t
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
[ recog_images, recog_result ] = interface_recognizeFish( video_id{56}, fish_detection{56} )
imshow(read(mov_properties, 287));
%-- 2013-08-16 03:13 PM --%
load('1.fish27k_attributes.mat', 'image_name')
for i = 1:27370
fish_image{i,1}=imread(image_name{i});
end
load('2.fish90k_fishImage.mat')
load('1.fish90k_attributes_20121121.mat', 'class_id')
mask_image = maskImg(class_id > 0 & class_id < 24);
for i = 1:27370
width(i,1)=size(fish_image{i},1);
height(i,1)=size(fish_image{i},2);
end
retry = find(width>200 | height > 200);
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
fish_image = fish_image(retry);
mask_image = mask_image(retry);
for i = 1:601
fish_image2{i,1}=imresize(fish_image{i,1}, 200/max(fish_image{i,1}));
mask_image2{i,1}=imresize(mask_image{i,1}, 200/max(mask_image{i,1}));
end
for i = 1:601
fish_image2{i,1}=imresize(fish_image{i,1}, 200/max(size(fish_image{i,1})));
mask_image2{i,1}=imresize(mask_image{i,1}, 200/max(size(mask_image{i,1})));
end
[features2]= interface_generateFeatureSet(fish_image2, mask_image2);
43*60/601
load('matlab.mat', 'features2')
features1 = features(retry,:);
sum(features1==features2)
for i = 1:601
tmp(i,1)=sum(features1(i,:)==features2(i,:));
end
for i = 1:601
tmp(i,1)=sum(features1(i,:)-features2(i,:));
end
sum(tmp)
for i = 1:601
tmp(i,1)=sum(features1(i,:)-features3(i,:));
end
features2 = features1;
features2(retry,:)=features3;
[features1]= interface_normalizeFeatureFromPrefix(features1, model_data.feature_mean, model_data.feature_std);
[result_predict1, result_scores1] = classify_HierSVM_predict(features1, model_data.BGOT_Trained);
if iscell(result_scores1)
result_scores1 = cell2mat(result_scores1);
end
[ result_predict1 ] = append_convertlabel( model_data.convert2database, result_predict1);
model_data = load('data_15Species.mat');
[features1]= interface_normalizeFeatureFromPrefix(features1, model_data.feature_mean, model_data.feature_std);
[result_predict1, result_scores1] = classify_HierSVM_predict(features1, model_data.BGOT_Trained);
if iscell(result_scores1)
result_scores1 = cell2mat(result_scores1);
end
[ result_predict1 ] = append_convertlabel( model_data.convert2database, result_predict1);
[features2]= interface_normalizeFeatureFromPrefix(features2, model_data.feature_mean, model_data.feature_std);
[result_predict2, result_scores2] = classify_HierSVM_predict(features2, model_data.BGOT_Trained);
if iscell(result_scores2)
result_scores2 = cell2mat(result_scores2);
end
[ result_predict2 ] = append_convertlabel( model_data.convert2database, result_predict2);
load('1.speces_15_cv5_convert.mat', 'class_id')
result1 = result_evaluate(result_predict1, class_id);
load('2.speces_23_cv5_convert.mat', 'class_id')
result1 = result_evaluate(result_predict1, class_id);
result1 = result_evaluate(result_predict1, class_id)
result2 = result_evaluate(result_predict2, class_id)
recall = result1.classrecall;
recall(:,2) = result2.classrecall;
recall = result1.classrecall';
recall(:,2) = result2.classrecall';
mean(recall(:,1))
mean(result(:,1))
mean(result(:,2))
%-- 2013-08-19 06:45 PM --%
load('1.video_id.mat')
interface_recognizeVideo(video_id(16,30),0,0);
interface_recognizeVideo(video_id(16:30),0,0);
%-- 2013-08-19 06:46 PM --%
load('1.video_id.mat')
interface_recognizeVideo(video_id([31:40,51:55]),0,0);
%-- 2013-08-19 06:47 PM --%
load('1.video_id.mat')
interface_recognizeVideo(video_id(56:70),0,0);
%-- 2013-08-19 06:52 PM --%
load('1.fish27k_attributes.mat', 'class_id_15')
tabulate(class_id_15)
a = tabulate(class_id_15)
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'traj_id')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'featurei')
indic = class_id_15 > 0;
features = features(indic,:);
traj_id = traj_id(indic,:);
class_id = class_id_15(indic,:);
load('9.fish7k2_result_tree4_data.mat', 'hier1')
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0, hier1, 0, 1);
%-- 2013-08-19 07:06 PM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'traj_id')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'featurei')
indic = class_id_15 > 0;
features = features(indic,:);
traj_id = traj_id(indic,:);
class_id = class_id_15(indic,:);
load('9.fish7k2_result_tree4_data.mat', 'hier1')
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0, hier1, 0, 1);
load('1.fish27k_attributes.mat', 'class_id_15')
indic = class_id_15 > 0;
features = features(indic,:);
traj_id = traj_id(indic,:);
class_id = class_id_15(indic,:);
load('9.fish7k2_result_tree4_data.mat', 'hier1')
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0, hier1, 0, 1);
[  result_hier_Traj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 1, hier1, 0, 1);
load('1.fish27k_attributes.mat', 'cv_id_fold5')
load('1.fish27k_attributes.mat', 'class_id_15')
cv_id = cv_id_fold5(class_id_15>0);
load('data_15Species.mat', 'feature_mean', 'feature_std')
[features]= interface_normalizeFeatureFromPrefix(features, feature_mean,feature_std);
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0, 0, 0, 1);
[  result_flat_noTraj] = classify_crossValidation(features, featurei, classid, traj_id, cv_id, 2, 0, 0, 1)
[  result_hier_noTraj] = classify_crossValidation(features, featurei, classid, traj_id, cv_id, 2, 0, hier1, 1)
[  result_flat_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id, 2, 0, 0, 1)
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id, 2, 0, hier1, 1)
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 0, hier1, 0, 1);
[  result_hier_noTraj] = classify_crossValidation(features, featurei, class_id, traj_id, 5, 2, 1, hier1, 0, 1);
clear;
%-- 2013-08-21 07:10 PM --%
%-- 2013-08-21 07:11 PM --%
load('1.video_id.mat', 'video_id')
interface_recognizeVideo(video_id(1:15),0,0,1,0);
%-- 2013-08-21 07:12 PM --%
load('1.video_id.mat', 'video_id')
interface_recognizeVideo(video_id(16:30),0,0,1,0);
interface_recognizeVideo(video_id(23:30),0,0,1,0);
interface_recognizeVideo(video_id(29:30),0,0,1,0);
interface_recognizeVideo(video_id(28:30),0,0,1,0);
%-- 2013-08-23 12:24 AM --%
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'class_id')
features = features(ismember([1,21], class_id),:);
index = ismember(class_id, [1,21]);
features = features(indic,:);
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features')
features = features(indic,:);
%-- 2013-08-25 10:57 PM --%
load('1.fish27k_attributes.mat', 'class_id_15')
tabulate(class_id_15)
sum(class_id_15)>0
sum(class_id_15>0)
load('1.fish27k_attributes.mat', 'traj_id')
unique(traj_id(class_id_15>0))
clear;
load('1.fish27k_attributes.mat', 'class_id_15')
indic = find(class_id_15 > 0);
class_id = class_id_15(indic);
load('1.fish27k_attributes.mat', 'fish_id')
load('1.fish27k_attributes.mat', 'traj_id')
traj_id = traj_id(indic);
load('1.fish27k_attributes.mat', 'cv_id_fold5')
cv_id_fold5 = cv_id_fold5(indic);
cv_id_fold5(class_id==11)
cv_id_fold5(class_id==13)
load('2.fish90k_feature_23species_130211_unnormalize.mat', 'features', 'featurei')
features = features(indic, :);
features2 = single(features);
de2bi(1, 15)
de2bi(1024, 15)
2^10
de2bi(2048, 15)
de2bi(4096, 15)
de2bi(9192, 15)
de2bi(8192, 15)
de2bi(16384, 15)
de2bi(18383, 15)
de2bi(16383, 15)
2^14
de2bi(0, 15)
for i = 0:16383
binary_pattern(i,1) = sum(de2bi(i, 15) == 1);
end
for i = 1:16383
binary_pattern(i,1) = sum(de2bi(i, 15) == 1);
end
floor(15 / 2)
sum(binary_pattern==7)
tmp = sum(binary_pattern==7);
tmp / 4
tmp(858)
tmp = find(binary_pattern==7);
tmp(858)
tmp(1716)
tmp(2574)
tmp(3432)
[ result1 ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, 1:15, 1, 4555, '0.split1_1.mat');
%-- 2013-08-25 11:46 PM --%
load('1.attributes.mat')
load('2.feature_unnormalized.mat')
%-- 2013-08-25 11:47 PM --%
load('1.attributes.mat')
load('2.feature_unnormalized.mat')
%-- 2013-08-25 11:47 PM --%
load('1.attributes.mat')
load('2.feature_unnormalized.mat')
[ result1 ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, 1:15, 11827, 16256, '0.split1_4.mat');
%-- 2013-08-25 11:49 PM --%
load('1.attributes.mat')
load('2.feature_unnormalized.mat')
[ result1 ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, 1:15, 1, 4555, '0.split1_1.mat');
%-- 2013-08-26 11:45 AM --%
load('1.attributes.mat')
load('2.feature_unnormalized.mat')
load('data_23Species.mat', 'Feature_model')
load('2.fish90k_feature_23species_130407_prefixNormalize.mat', 'prefix_mean', 'prefix_std')
load('2.fish90k_feature_23species_130407_prefixNormalize95.mat', 'prefixValue')
[features_BGOT]= interface_normalizeFeatureFromPrefix(features, Feature_model.feature_mean, Feature_model.feature_std);
[features_self, prefixValue]= interface_normalizeFeatureSet(features);
(Feature_model.feature_mean-prefixValue.mean_value)./Feature_model.feature_mean
(Feature_model.feature_mean'-prefixValue.mean_value)./Feature_model.feature_mean
(Feature_model.feature_mean-prefixValue.mean_value')./Feature_model.feature_mean
tmp = abs(tmp);
tmp = tmp(~isnan(tmp));
find(tmp>0.3)
tmp(tmp>0.3)
tmp(tmp>0.5)
tmp(tmp>0.3)
tmp(tmp>0.4)
tmp(tmp>0.5)
tmp(tmp>0.6)
tmp>0.6
find(tmp>0.6)
(Feature_model.feature_mean-prefixValue.mean_value')./Feature_model.feature_mean
find(tmp>0.6)
tmp(tmp>0.6)
find(tmp>0.6)
tmp=(Feature_model.feature_std-prefixValue.mean_std')./(Feature_model.feature_std+prefixValue.mean_std')
tmp=(Feature_model.feature_std-prefixValue.std_value')./(Feature_model.feature_std+prefixValue.std_value')
find(tmp>0.3)
prefixValue.std_value(2047)
find(tmp>0.1)
find(abs(tmp)>0.1)
find(abs(tmp)>0.3)
tmp=(Feature_model.feature_mean-prefixValue.mean_value')./(Feature_model.feature_mean+prefixValue.mean_value')
tmp = abs(tmp);
find(abs(tmp)>0.3)
find(abs(tmp)>0.6)
load('2.feature_unnormalized.mat', 'features')
[features_self, prefixValue]= interface_normalizeFeatureSet(features);
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
%-- 2013-08-26 12:13 PM --%
load('2.feature_unnormalized.mat')
[features_self, prefixValue]= interface_normalizeFeatureSet(features);
load('2.feature_unnormalized.mat', 'features')
load('1.attributes.mat')
load('data_23Species.mat', 'Feature_model')
[features_prefix]= interface_normalizeFeatureFromPrefix(features, prefixValue.mean_value, prefixValue.std_value);
[features_prefix]= interface_normalizeFeatureFromPrefix(features, prefixValue.mean_value', prefixValue.std_value');
[features_self, prefixValue]= interface_normalizeFeatureSet(features);
[features_prefix]= interface_normalizeFeatureSet(features);
[  result_prefix] = classify_crossValidation(features_prefix, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
[  result_self] = classify_crossValidation(features_self, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
%-- 2013-08-26 12:48 PM --%
load('2.feature_unnormalized.mat', 'features')
[features_self, prefixValue]= interface_normalizeFeatureSet(features);
load('2.feature_unnormalized.mat', 'features')
[features_10std]= interface_normalizeFeatureSet(features);
load('1.attributes.mat')
load('2.feature_unnormalized.mat', 'featurei')
[  result] = classify_crossValidation(features_10std, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
tabulate(classid_train)
[  result] = classify_crossValidation(features_no10std, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
%-- 2013-08-26 01:35 PM --%
load('2.feature_unnormalized.mat')
load('1.attributes.mat')
[features_10std]= interface_normalizeFeatureSet(features);
[features_95bound]= interface_normalizeFeatureSet(features);
[features_10std]= interface_normalizeFeatureSet(features);
[features_95bound]= interface_normalizeFeatureSet(features);
[features_95bound2]= interface_normalizeFeatureSet(features);
sum(features_95bound == features_95bound2)
sum(abs(features_95bound - features_95bound2)<0.001)
[features_10std2]= interface_normalizeFeatureSet(features);
sum(abs(features_10std - features_10std2)<0.001)
sum(sum(abs(features_10std - features_10std2)<0.001)<10000)
[features_10std2]= interface_normalizeFeatureSet(features);
sum(abs(features_10std - features_10std2)<0.001)
[features_20std]= interface_normalizeFeatureSet(features);
[features_10std]= interface_normalizeFeatureSet(features);
[  result] = classify_crossValidation(features_10std, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
[  result] = classify_crossValidation(features_10std_exclude5, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
%-- 2013-08-26 02:10 PM --%
load('matlab_20std.mat')
%-- 2013-08-26 02:10 PM --%
load('matlab_95bound.mat')
[  result] = classify_crossValidation(features_95bound, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
load('2.feature_unnormalized.mat', 'features')
means = mean(featuers);
means = mean(features);
stds = std(features);
stds = std(features)+eps;
[features_nobound_zscore]= interface_normalizeFeatureFromPrefix(features, means, stdd)
[features_nobound_zscore]= interface_normalizeFeatureFromPrefix(features, means, stds);
[  result] = classify_crossValidation(features_nobound_zscore, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
load('data_23Species.mat', 'Feature_model')
%-- 2013-08-26 03:26 PM --%
load('2.feature_unnormalized.mat')
load('1.attributes.mat')
[features, prefix]= interface_normalizeFeatureSet(features);
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
%-- 2013-08-26 03:29 PM --%
load('2.feature_normalized_self95.mat')
load('1.attributes.mat')
features = single(features);
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
features = double(features);
[  result] = classify_crossValidation(features, featurei, class_id, traj_id, cv_id_fold5, 2, 0, 0, 0)
clear;
load('2.feature_unnormalized.mat')
features = single(features);
features = float(features);
load('2.feature_unnormalized.mat')
features = single(features);
clear;
load('2.feature_normalized_self95.mat')
features = single(features);
clear;
load('1.attributes.mat')
load('2.feature_normalized_self95.mat')
features = double(features);
[ result2 ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, 1:15, 4556, 8128, '0.split1_2.mat');
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, cv_id_fold5, 2, 'fs_root_69', 0, 0)
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, cv_id_fold5, 2, 'fs_root_individual', 0, 0)
%-- 2013-09-14 10:02 PM --%
load('matlab.mat')
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, cv_id_fold5, 2, 'fs_root_individual', 0, 0)
load('5.fs_root_69.mat', 'scoreResult')
%-- 2013-09-18 02:24 PM --%
%-- 2013-09-18 06:41 PM --%
load('2.feature_normalized_self95.mat', 'features', 'featurei')
load('1.attributes.mat', 'class_id', 'traj_id', 'cv_id_fold5')
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, cv_id_fold5, 2, 'fs_root_individual', 0, 0)
features = double(features);
[featureSubset, scoreResult] = classify_featureselection_fw(features, 1:2626, class_id, traj_id, 500, 3, cv_id_fold5, 2, 'fs_root_individual', 0, 0)
clear;
load('6.fs_root_individual.mat')
%-- 2013-09-28 03:18 AM --%
