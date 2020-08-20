function [ hier_tree, resultTable ] = append_hier_assignFSSubset( hier_tree, featurei, id_set, filePattern, fileRoot )

resultTable = {};
if isempty(hier_tree)
    return;
end

if ~exist('filePattern', 'var')
   filePattern = 'featureSelection_Node_%02d.mat';
end

if ~exist('featurei', 'var')
   featurei = [];
end

if ~exist('id_set', 'var')
   id_set = [];
end

if ~exist('fileRoot', 'var')
   fileRoot = 'featureSelection_Root.mat';
end

[ hier_tree, resultTable ] = inner_assignNodeAndChildren_FSSubset( hier_tree, featurei, id_set, filePattern, resultTable );

if exist(fileRoot, 'file') || exist([fileRoot, 'mat'], 'file')
    tmp_data = load(fileRoot);
    [ resultTable, hier_tree.Root.Subfeature ] = inner_findBestSubset( size(resultTable,1) + 1, tmp_data.scoreResult, resultTable, tmp_data.featureSubset, featurei );
end
end

function [ hier_tree, resultTable ] = inner_assignNodeAndChildren_FSSubset( hier_tree, featurei, id_set, filePattern, resultTable )

if isempty(hier_tree)
    return;
end

node_id = hier_tree.node_id;
fileName = sprintf(filePattern, node_id);
if exist(fileName, 'file') || exist([fileName, 'mat'], 'file')
    if isempty(id_set) | find(id_set==node_id)
        tmp_data = load(fileName);
        
        [ resultTable, hier_tree.Subfeature ] = inner_findBestSubset( node_id, tmp_data.scoreResult, resultTable, tmp_data.featureSubset, featurei );
        %    if resultTable{node_id, 2} == length(tmp_data.featureSubset)
        %        hier_tree = rmfield(hier_tree, 'Subfeature');
        %    end
        %
        %    if  resultTable{node_id, 2} < 5
        %        hier_tree = rmfield(hier_tree, 'Subfeature');
        %    end
    end
    
end

for i = 1:length(hier_tree.branch_link)
    [ hier_tree.branch_link{i}, resultTable ] = inner_assignNodeAndChildren_FSSubset( hier_tree.branch_link{i}, featurei, id_set, filePattern, resultTable );
end

end

function [ resultTable, Subfeature ] = inner_findBestSubset( node_id, scoreResult, resultTable, featureSubset, featurei )
   [mv, mi] = max(scoreResult(:,3));
   
   resultTable{node_id, 1} = scoreResult(mi,:);
   resultTable{node_id, 2} = mi;
   resultTable{node_id, 3} = featureSubset;
   
   if isempty(featurei)
       Subfeature = featureSubset(1:mi);
   else
       Subfeature = find(ismember(featurei, featureSubset(1:mi)));
   end
end
