function [rt_predict, rt_gtclass, rt_scores, hier_arch] = classify_HierSVM(feature_train, classid_train, feature_test, classid_test, featurei, hier_arch, classifier, option_add)

%
% Xuan (Phoenix) Huang                Xuan.Huang@ed.ac.uk
% Edinburgh, United Kingdom                      Feb 2012 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%classify_HierSVM

if ~exist('option_add', 'var')
    option_add = '';
end

%hierarch classification for each node
[hier_arch] = classify_HierSVM_train(feature_train, classid_train, featurei, hier_arch, classifier, option_add);


[rt_predict, rt_scores, hier_arch] = classify_HierSVM_predict(feature_test, hier_arch);
[hier_arch] = classify_HierSVM_evaluate(hier_arch, classid_test);

rt_gtclass = classid_test;
end