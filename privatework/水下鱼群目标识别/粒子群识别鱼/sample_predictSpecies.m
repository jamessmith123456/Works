addpath('./fish_recog');

%load testing image data
tmp_data = load('image.mat');
input_rgbImage = tmp_data.rgbimg;
input_mskImage = tmp_data.mskimg;
traj_id = tmp_data.traj_id;

%load pre-trained model data
model_data = load('data_23Species.mat');

% call fish recognition interface
%
% prediction output label:
% 01	Dascyllus reticulatus (1)
% 02	Chromis margaritifer (3)
% 03	Plectroglyphidodon dickii (2)
% 04	Acanthurus (8)
% 05	Myripristis kuntee (7)
% 06	Chaetodon trifascialis (6)
% 07	Zebrasoma scopas (14)
% 08	Scolopsis bilineata (17)
% 09	Amphiprion clarkii (4)
% 10	Siganus fuscescens (23)
% 11	Pomacentrus moluccensis (13)
% 13	Scaridea (18)
% 17	Canthigaster valentini (12)
% 19	Balistapus undulatus (22)
% 21	Hemigymnus melapterus (15)
% 24	Hemigymnus fasciatus (9)
% 26	Abudefduf vaigiensis (11)
% 27	Lutjanus fulvus (16)
% 32	Chaetodon lunulates (5)
% 33	Neoniphon sammara (10)
% 34	Pempheris vanicolensis (19)
% 37	Neoglyphidodon nigroris (21)
% 38	Zanclus cornutus  (20)
%
% scores is a sample-number-length vector, each row is the BGOT score of given sample.
% rejected fish will be labeled as 0.
[ detection_predict, detection_score, traj_predict, traj_scores ] = interface_recognizeFishFromImage(model_data , input_rgbImage, input_mskImage,  traj_id);
