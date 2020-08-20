function [hier_arch] = classify_HierSVM_TreePredict(hier_arch)

%
% Xuan (Phoenix) Huang                Xuan.Huang@ed.ac.uk
% Edinburgh, United Kingdom                      Feb 2012 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%classify_HierSVM_TreePredict reconstructs hierarch result

number_node = length(hier_arch.branch_class);

%hier_arch.Tree_gtclass = hier_arch.Input_classid_test;
%hier_arch.Tree_gtclass(~ismember(hier_arch.Tree_gtclass, hier_arch.Input_classid_set))=-1;

hier_arch.Tree_predict = hier_arch.Node_predict;

%applied to child link first
for i = 1:number_node
    tmp_indic= hier_arch.Node_predict==i;
    if ~isempty(hier_arch.branch_link{i}) && sum(tmp_indic)
        hier_arch.branch_link{i} = classify_HierSVM_TreePredict(hier_arch.branch_link{i});
        
        hier_arch.Tree_predict(tmp_indic) = hier_arch.branch_link{i}.Tree_predict;
    end
end


%re-assign class name to single node
for i = 1:number_node
    if 1== length(hier_arch.branch_class{i})
        hier_arch.Tree_predict(hier_arch.Node_predict==i) = hier_arch.branch_class{i};
    end
end

% hier_arch.Tree_Result_details = zeros(3,1);
% hier_arch.Tree_Result_details(1,1)=-1;
% testcls_indicate = (hier_arch.Tree_gtclass == -1);
% hier_arch.Tree_Result_details(2,1)=sum(hier_arch.Tree_predict(testcls_indicate)==-1);
% hier_arch.Tree_Result_details(3,1)=sum(testcls_indicate);
% hier_arch.Tree_Result_details(4,1)=sum(hier_arch.Tree_predict == -1);

%add predict history
hier_arch.Tree_Predict_history = cell(length(hier_arch.Node_predict), 4);
tmp_node_id = 0;
if isfield(hier_arch, 'node_id')
	tmp_node_id = hier_arch.node_id;
end
for i = 1:length(hier_arch.Node_predict)
    hier_arch.Tree_Predict_history{i,1} = hier_arch.Tree_predict(i);
    hier_arch.Tree_Predict_history{i,2} = 0; %hier_arch.Tree_gtclass(i);
    hier_arch.Tree_Predict_history{i,3} = tmp_node_id;
    hier_arch.Tree_Predict_history{i,4} = hier_arch.Node_predict(i);
end
%end add predict history

%add score history
hier_arch.Tree_Score_history = hier_arch.Node_scores;
%end add score history

for i = 1:number_node
    
    if 0 < length(hier_arch.branch_link{i})
        tmp_predict_set = find(hier_arch.Node_predict == i);
        for j = 1:length(tmp_predict_set)
            %add predict history
            hier_arch.Tree_Predict_history{tmp_predict_set(j),3} = [hier_arch.Tree_Predict_history{tmp_predict_set(j),3}, hier_arch.branch_link{i}.Tree_Predict_history{j,3}];
            hier_arch.Tree_Predict_history{tmp_predict_set(j),4} = [hier_arch.Tree_Predict_history{tmp_predict_set(j),4}, hier_arch.branch_link{i}.Tree_Predict_history{j,4}];
            %end add predict history  
            
            %add score history
            tmp_add_score = hier_arch.branch_link{i}.Tree_Score_history(j,:); 
            tmp_add_size = find(cellfun(@isempty, tmp_add_score)==0, 1, 'last');
            tmp_insert_size = find(cellfun(@isempty, hier_arch.Tree_Score_history(tmp_predict_set(j),:))==0, 1, 'last');           
            hier_arch.Tree_Score_history(tmp_predict_set(j),tmp_insert_size+1:tmp_insert_size+tmp_add_size) = tmp_add_score(1:tmp_add_size);
            %end add score history
        end
        
    end
    
%     test_class_set = hier_arch.branch_class{i};
%     for j = 1:length(test_class_set)
%         tmp_result(1,1)=test_class_set(j);
%         tmp_result(2,1)=sum(hier_arch.Tree_predict(hier_arch.Tree_gtclass == test_class_set(j)) == test_class_set(j));
%         tmp_result(3,1)=sum(hier_arch.Tree_gtclass == test_class_set(j));
%         tmp_result(4,1)=sum(hier_arch.Tree_predict == test_class_set(j));
%         
%         hier_arch.Tree_Result_details(:,end+1)=tmp_result;
%     end
end

% warning off;
% tmpscore = hier_arch.Tree_Result_details(2,2:end) ./ hier_arch.Tree_Result_details(3,2:end);
% hier_arch.Result_Tree_clsscore = mean(tmpscore(~isnan(tmpscore)));
% warning on;


end