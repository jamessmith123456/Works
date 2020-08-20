addpath('./fish_recog');

%load testing image data
tmp_data = load('features.mat');

[ hier ] = interface_trainBGOTFromImage( tmp_data.features, tmp_data.class_id );
