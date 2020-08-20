function [ result ] = script_NodeSplit_N2_reGroup( id )

if ischar(id)
    id = str2num(id);
end

total = 128;

%data_file = 'E:\Documents\NetStorage\Dropbox\2011-01-01_Matlab_data\2012-10-30_fish_90k\2.fish90k_feature_23species_130226_normalize.mat';
%data_file = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2012-10-30_fish_90k\2.fish90k_feature_23species_130226_normalize.mat';
data_file = '/exports/work/inf_ipab/phoenix/data/2.fish90k_feature_23species_130226_nor.mat';

load(data_file, 'features', 'cv_id_fold5', 'class_id', 'traj_id');

branch_class=[1,2,3,7,8,9,14,15,17,18,21,22];
[ features, class_id, traj_id, cv_id_fold5 ] = append_filteClass( branch_class, features, class_id, traj_id, cv_id_fold5 );

class_set = 1:12;
featurei = 1:2626;
fprintf('id: %d, total: %d\n', id, total);

range_whole = 2^(length(class_set)-1);
range_block = ceil(range_whole/total);
range_start = range_block * (id - 1) + 1;
range_end = range_block * id;
filesavename = sprintf('nodeSplit_%04d_%d.mat', id, total);

[ result ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id_fold5, class_set, range_start, range_end, filesavename);

end

