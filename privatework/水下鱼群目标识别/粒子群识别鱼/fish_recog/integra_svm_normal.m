function [predictClass result] = integra_svm_normal()

feature_train = feature_normal(fish_trainset,:);
feature_test = feature_normal(fish_testset,:);
classid_train = [fish_name(fish_trainset).classid];
classid_test = [fish_name(fish_testset).classid];
[model_w model_b] = classify_trainSVM(feature_train, classid_train);
[predictClass result] = result_evaluate(feature_test, classid_test, model_w, model_b);
errorbar(result.classaverageset', zeros(11,1), 'b+');

end