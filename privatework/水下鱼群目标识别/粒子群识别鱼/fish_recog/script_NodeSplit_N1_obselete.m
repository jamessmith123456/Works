function [ result ] = script_NodeSplit_N1( id )

if ischar(id)
    id = str2num(id);
end

total = 100;

data_file = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2013-09-28_fish27k_new16\1.features_self95.mat';
%data_file = '/exports/work/inf_ipab/phoenix/data/1.features_self95.mat';

load(data_file, 'features', 'cv_id', 'featurei', 'class_id', 'traj_id');
class_set = sort(unique(class_id));
fprintf('id: %d, total: %d\n', id, total);

range_whole = 2^length(class_set);
range_block = ceil(range_whole/total);
range_start = range_block * (id - 1) + 1;
range_end = range_block * id;
filesavename = sprintf('nodeSplit_%04d_%d.mat', id, total);

[ result ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id, class_set, range_start, range_end, filesavename);

end

