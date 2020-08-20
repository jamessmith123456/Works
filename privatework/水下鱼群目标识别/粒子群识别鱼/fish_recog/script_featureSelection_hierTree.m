function script_featureSelection_hierTree( id, boolFeatureType, boolMinus1 )

if ischar(id)
    id = str2num(id);
end

if ischar(boolFeatureType)
    boolFeatureType = str2num(boolFeatureType);
end

if ischar(boolMinus1)
    boolMinus1 = str2num(boolMinus1);
end


data_path = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2013-08-25_fish24k\';
%data_path = '/exports/work/inf_ipab/phoenix/data/';

feature_file = '2.feature_normalized_self95.mat';
load([data_path feature_file], 'features', 'featurei');
attribute_file = '1.attributes.mat';
load([data_path attribute_file], 'class_id', 'traj_id', 'cv_id_fold5');
tree_file = '4.fish24k_hierTree_1301.mat';
load([data_path tree_file], 'hierTree');
features = double(features);
[samples, dimens] = size(features);


if ~boolFeatureType
    fprintf('feature selection on individual features\n');
    featurei = 1:dimens;
else
    fprintf('feature selection on %d types features\n', length(unique(featurei)));
end


hierNode = inner_findNode(hierTree, id);

filesavename = sprintf('fs_Node_%02d_type%d_minus%d.mat', id, boolFeatureType, boolMinus1);

[ features, class_id, traj_id, cv_id_fold5 ] = append_filteClassGroup( hierNode.branch_class, features, class_id, traj_id, cv_id_fold5, boolMinus1 );

classify_featureselection_fw(features, featurei, class_id, traj_id, length(unique(featurei)), 3, cv_id_fold5, 2, filesavename, 0, 0);

end

function [hierNode] = inner_findNode(hierTree, id)
hierNode = [];
if isempty(hierTree) || hierTree.node_id == id
    hierNode =  hierTree;
    return;
end

for i = 1:length(hierTree.branch_link)
    [hierNode] = inner_findNode(hierTree.branch_link{i}, id);
    if ~isempty(hierNode)
        return;
    end
end
return;
end
