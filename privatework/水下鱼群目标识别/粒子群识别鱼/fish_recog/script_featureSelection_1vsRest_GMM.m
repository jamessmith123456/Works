function script_featureSelection_1vsRest_GMM( species_id,  boolFeatureType)

if ischar(species_id)
    species_id = str2num(species_id);
end


if ischar(boolFeatureType)
    boolFeatureType = str2num(boolFeatureType);
end


data_path = 'C:\Envrion\Dropbox\2011-01-01_Matlab_data\2013-08-25_fish24k\';
%data_path = '/exports/work/inf_ipab/phoenix/data/';

feature_file = '2.feature_normalized_self95.mat';
load([data_path feature_file], 'features', 'featurei');
attribute_file = '1.attributes.mat';
load([data_path attribute_file], 'class_id', 'traj_id', 'cv_id_fold5', 'split_root_range100');
features = double(features);
[samples, dimens] = size(features);

if ~boolFeatureType
    fprintf('feature selection on individual features\n');
    featurei = 1:dimens;
else
    fprintf('feature selection on %d types features\n', length(unique(featurei)));
end

%featurei = 1:dimens;
fprintf('species id: %d\n', species_id);

filesavename = sprintf('data_fs_GMM_Species_%04d_type%d', species_id, boolFeatureType);
classify_GMM_fs_fw(features, featurei, class_id, species_id, length(unique(featurei)), cv_id_fold5, filesavename)
end

