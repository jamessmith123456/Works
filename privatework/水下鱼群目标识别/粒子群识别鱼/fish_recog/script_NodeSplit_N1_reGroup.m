function [ result ] = script_NodeSplit_N1_reGroup( id )

if ischar(id)
    id = str2num(id);
end

total = 128;

%data_file = 'E:\Documents\NetStorage\Dropbox\2011-01-01_Matlab_data\2012-10-30_fish_90k\2.fish90k_feature_23species_130226_normalize.mat';
%data_file = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2012-10-30_fish_90k\2.fish90k_feature_23species_130226_normalize.mat';
data_file = '/exports/work/inf_ipab/phoenix/data/2.fish90k_feature_23species_130226_nor.mat';

load(data_file, 'features', 'cv_id_fold5', 'class_id', 'traj_id');

convert_index_27k = [1,4,9,8,7,7,5,2,3,10,8,6,11,4,3,10,5,2,11,6,9,1,12];
class_id_group_27k = zeros(6874,1);
for tmp_i = 1:23
    class_id_group_27k(class_id==tmp_i) = convert_index_27k(tmp_i);
end

class_set = 1:12;
featurei = 1:2626;
fprintf('id: %d, total: %d\n', id, total);

range_whole = 2^(length(class_set)-1);
range_block = ceil(range_whole/total);
range_start = range_block * (id - 1) + 1;
range_end = range_block * id;
filesavename = sprintf('nodeSplit_%04d_%d.mat', id, total);

[ result ] = append_testNodeSplit( features, featurei, class_id_group_27k, traj_id, cv_id_fold5, class_set, range_start, range_end, filesavename);

end

