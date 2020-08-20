function [ detection_predict, detection_score, traj_predict, traj_scores, traj_array ] = interface_recognizeFishFromImage( model_data, rgbImg, binImg, traj_id )
%
% 
% Usage syntax:
% [ detection_predict, detection_score, traj_predict, traj_scores, traj_array ] = interface_recognizeFishFromImage( model_data, rgbImg, binImg, traj_id )
%
% Inputs:
%
% "model_data" is the BGOT model; it is a pre-trained hierarchical
%   classification tree. A BGOT of 15 common fish in Taiwan sea is provided.
% "rgbImg", "binImg" are the input images of fish and mask. 
%
% "traj_id" is the tracking id which is used to eliminate errors along
%   a whole trajectory.
%
% Outputs:
%
% "detection_predict" is a sample-number-length vector, each row is the 
%   BGOT prediction of given sample. See the label table below.
% "detection_score" is a sample-number-length vector, each row is the BGOT
%   score of given sample. The rejected fish will be labeled as 0.
% "traj_predict" is a trajectory-number-length vector, each row is the 
%   BGOT prediction of given trajectory. See the label table below.
% "traj_scores" is a trajectory-number-length vector, each row is the BGOT
%   score of given trajectory (at most 3). The rejected fish will be labeled as 0.
% "traj_array" is a trajectory-number-length vector, 
%   each row is the trajectory id of each 

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
% Written by:
%   Phoenix X. Huang
%   University of Edinburgh
%   U.K.
%
%   Email: Xuan.Huang@ed.ac.uk
%
%   2011, 2012, 2013
% 
% -----------------------------------------------------------------------
% Copyright (2013): Phoenix X. Huang
%
% This software is distributed under the terms
% of the GNU General Public License 2.0.
% 
% Permission to use, copy,  and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
if length(rgbImg) ~= length(binImg)
    return;
end

if ~exist('traj_id', 'var')
    traj_id = 1:length(rgbImg);
end

% rejected fish will be labeled as 0.
[detection_predict, detection_score, traj_predict, traj_scores, traj_array ] = interface_prediction( rgbImg, binImg, model_data, traj_id );

end