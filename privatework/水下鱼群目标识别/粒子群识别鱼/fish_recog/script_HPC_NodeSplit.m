function [ result ] = script_HPC_NodeSplit( id )

if ischar(id)
    id = str2num(id);
end

total = 429;

data_path = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2013-09-28_fish27k_new16\1.features_self95.mat';
%data_path = '/exports/work/inf_ipab/phoenix/data/1.features_self95.mat';

load(data_path, 'features', 'featurei', 'class_id', 'traj_id', 'cv_id', 'split_root_range429');
features = double(features);
class_set = sort(unique(class_id));
fprintf('id: %d, total: %d\n', id, total);

range_start = split_root_range429(id,1);
range_end = split_root_range429(id,2);
filesavename = sprintf('nodeSplit_%04d_%d.mat', id, total);

time_start = clock();
time_tic = tic;
[ result ] = append_testNodeSplit( features, featurei, class_id, traj_id, cv_id, class_set, range_start, range_end, filesavename);
time_toc = toc(time_tic);
time_end = clock();

save(filesavename, 'time_tic', 'time_toc', 'time_start', 'time_end', '-append');

end

