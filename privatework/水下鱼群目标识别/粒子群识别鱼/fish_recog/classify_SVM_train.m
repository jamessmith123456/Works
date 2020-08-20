function [model] = classify_SVM_train(feature_train, classid_train, modeloption, option_add)

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~isa(feature_train,'double')
    feature_train = double(feature_train);
end

if nargin < 3
    modeloption = 2;
end
%1 libsvm 1 vs rest
%2 libsvm 1 vs 1
%3 matlabsvm 1vsRest

if ~exist('option_add', 'var')
    option_add = '';
end
libsvm_options_train = ['-s 0 -q -t 0 -c 1 -b 0 ', option_add];
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

%labelSet = unique(classid_train);
[B,I] = unique(classid_train, 'first');
labelSet = classid_train(sort(I));
if 1 == modeloption
    labelSetSize = length(labelSet);
    models = cell(labelSetSize,1);

    for i=1:labelSetSize
        models{i} = libsvmtrain(double(classid_train == labelSet(i)), feature_train, libsvm_options_train);
    end

    model.models = models;
elseif 2 == modeloption
    model.models = libsvmtrain(classid_train, feature_train, libsvm_options_train);
    
    [B,I] = unique(classid_train, 'first');
    model.classid_array = classid_train(sort(I));
    
    for i = 1:length(model.classid_array)
        model.class_prior(i) = sum(classid_train == model.classid_array(i)) / length(classid_train);
    end
elseif 3 == modeloption
    labelSetSize = length(labelSet);
    models = cell(labelSetSize,1);

    for i=1:labelSetSize
        models{i} = svmtrain(feature_train, double(classid_train ~= labelSet(i)));
    end

    model.models = models;
end

model.labelSet = labelSet;
model.modeloption = modeloption;
end