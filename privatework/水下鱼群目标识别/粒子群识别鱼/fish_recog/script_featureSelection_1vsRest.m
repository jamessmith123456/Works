function script_featureSelection_1vsRest( species_id )

if ischar(species_id)
    species_id = str2num(species_id);
end

%data_path = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2013-08-25_fish24k\';
data_path = '/exports/work/inf_ipab/phoenix/data/';

feature_file = '2.feature_normalized_self95.mat';
load([data_path feature_file], 'features', 'featurei');
attribute_file = '1.attributes.mat';
load([data_path attribute_file], 'class_id', 'traj_id', 'cv_id_fold5', 'split_root_range100');
features = double(features);
[samples, dimens] = size(features);

%featurei = 1:dimens;
fprintf('species id: %d\n', species_id);
binary_class = 2*ones(samples, 1);
binary_class(class_id == species_id) = 1;

filesavename = sprintf('data_featureSelection_Species_%04d', species_id);

classifier = 2;
standard = 3;
classify_featureselection_fw(features, featurei, binary_class, traj_id, dimens, standard, cv_id_fold5, classifier, filesavename, 0, 0);

end

