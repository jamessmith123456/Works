function [predict, scores, model] = classify_SVM(feature_train, classid_train, feature_test, modeloption, option_add)
if ~exist('option_add', 'var')
    option_add = '';
end

[model] = classify_SVM_train(feature_train, classid_train, modeloption, option_add);
[predict, scores] = classify_SVM_predict(model, feature_test);
end