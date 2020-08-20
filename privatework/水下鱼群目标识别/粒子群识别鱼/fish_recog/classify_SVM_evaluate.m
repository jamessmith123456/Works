function [predict, classid_test, scores, accuracy] = classify_SVM_predict(model, feature_test, classid_test)

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

modeloption = model.modeloption;

obj_num = size(feature_test, 1);
if nargin < 3
    classid_test = ones(obj_num,1);
end

breaking_tie = 0;
%0: do nothing
%1: use prior number
%2: use accumulate score

libsvm_options_test = '-b 0';
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_set_file)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% -v n: n-fold cross validation mode
% -q : quiet mode (no outputs)

if 1 == modeloption
    labelSet = model.labelSet;
    labelSetSize = length(labelSet);
    models = model.models;
    scores= zeros(size(classid_test, 1), labelSetSize);

    for i=1:labelSetSize
        [l,a,d] = svmpredict(double(classid_test == labelSet(i)), feature_test, models{i});
        scores(:, i) = d * (2 * models{i}.Label(1) - 1);
    end
    [tmp,pred] = max(scores, [], 2);
    predict = labelSet(pred);
    accuracy = sum(classid_test==pred) / size(feature_test, 1);
elseif 1 < modeloption
    [predict, accuracy, scores] = svmpredict(classid_test, feature_test, model.models, libsvm_options_test);
    classid_train = model.classid_train;
    
    [B,I] = unique(classid_train, 'first');
    classid_array = classid_train(sort(I));
    class_prior = zeros(1, length(classid_array));
    if 1 == breaking_tie
        for i = 1:length(classid_array)
            class_prior(i) = sum(classid_train == classid_array(i)) / length(classid_train);
        end
        [predict, scores] = result_1vs1( model.models.nr_class, scores, classid_array, class_prior );
    elseif 0 == breaking_tie
        [predict, scores] = result_1vs1( model.models.nr_class, scores, classid_array, class_prior );
    elseif 2 == breaking_tie
        [predict, scores] = result_1vs1( model.models.nr_class, scores, classid_array );
    end

    
end

scores = mat2cell(scores, ones(size(scores, 1),1), size(scores, 2));

end