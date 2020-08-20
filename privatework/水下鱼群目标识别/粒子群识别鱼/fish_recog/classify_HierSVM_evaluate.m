function [hier_arch] = classify_HierSVM_evaluate(hier_arch, classid_test)

%
% Xuan (Phoenix) Huang                Xuan.Huang@ed.ac.uk
% Edinburgh, United Kingdom                      Feb 2012 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%classify_HierSVM_TreePredict reconstructs hierarch result

number_node = length(hier_arch.branch_class);

number_test = length(classid_test);
Node_classid_test = -1 * ones(number_test, 1);
for i = 1:number_node
    Node_classid_test(ismember(classid_test, hier_arch.branch_class{i})) = i;
end
hier_arch.Node_classid_test = Node_classid_test;

hier_arch.Node_Result_details = zeros(3,1);
hier_arch.Node_Result_details(1,1)=-1;
hier_arch.Node_Result_details(2,1)=sum(hier_arch.Node_predict(hier_arch.Node_classid_test == -1)==-1);
hier_arch.Node_Result_details(3,1)=sum(hier_arch.Node_classid_test == -1);
hier_arch.Node_Result_details(4,1)=sum(hier_arch.Node_predict == -1);

for i = 1:number_node
    tmp_result(1,1)=i;
    tmp_result(2,1)=sum(hier_arch.Node_predict(hier_arch.Node_classid_test == i)==i);
    tmp_result(3,1)=sum(hier_arch.Node_classid_test == i);
    tmp_result(4,1)=sum(hier_arch.Node_predict == i);

    hier_arch.Node_Result_details(:,end+1)=tmp_result;
end

warning off;
tmpscore = hier_arch.Node_Result_details(2,2:end) ./ hier_arch.Node_Result_details(3,2:end);
hier_arch.Node_Result_scores = mean(tmpscore(~isnan(tmpscore)));
warning on;

hier_arch.Tree_gtclass = classid_test;
hier_arch.Tree_gtclass(~ismember(hier_arch.Tree_gtclass, hier_arch.Input_classid_set))=-1;

%applied to child link first
for i = 1:number_node
    tmp_indic= hier_arch.Node_predict==i;
    if ~isempty(hier_arch.branch_link{i}) && sum(tmp_indic)
        hier_arch.branch_link{i} = classify_HierSVM_evaluate(hier_arch.branch_link{i}, classid_test(tmp_indic));
    end
end

hier_arch.Tree_Result_details = zeros(3,1);
hier_arch.Tree_Result_details(1,1)=-1;
testcls_indicate = (hier_arch.Tree_gtclass == -1);
hier_arch.Tree_Result_details(2,1)=sum(hier_arch.Tree_predict(testcls_indicate)==-1);
hier_arch.Tree_Result_details(3,1)=sum(testcls_indicate);
hier_arch.Tree_Result_details(4,1)=sum(hier_arch.Tree_predict == -1);

for i = 1:number_node
    
    test_class_set = hier_arch.branch_class{i};
    for j = 1:length(test_class_set)
        tmp_result(1,1)=test_class_set(j);
        tmp_result(2,1)=sum(hier_arch.Tree_predict(hier_arch.Tree_gtclass == test_class_set(j)) == test_class_set(j));
        tmp_result(3,1)=sum(hier_arch.Tree_gtclass == test_class_set(j));
        tmp_result(4,1)=sum(hier_arch.Tree_predict == test_class_set(j));
        
        hier_arch.Tree_Result_details(:,end+1)=tmp_result;
    end
end

warning off;
tmpscore = hier_arch.Tree_Result_details(2,2:end) ./ hier_arch.Tree_Result_details(3,2:end);
hier_arch.Result_Tree_clsscore = mean(tmpscore(~isnan(tmpscore)));
warning on;


end