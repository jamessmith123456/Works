function [ BGOT ] = interface_trainBGOTFromImage_update( features, class_id )
%
% 
% Usage syntax:
% [ model_data ] = interface_trainBGOTFromImage( rgbImg, binImg, class_id )
%
% Inputs:
%
% "features" is the input features of fish. 
%
% "class_id" is the class label.
%
% Outputs:
%
% "model_data" is the BGOT model; it is a pre-trained hierarchical
%   classification tree.


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

%set default parameters
[samples, dimens] = size(features);
featurei = 1:dimens; %feature id
traj = 1:samples; %trajectory id
fold=5; %5 fold cross validation
classifier=2; %use 1vs1 classifier
traj_vote=0; %not use trajectory voting
class_set=unique(class_id); %input class set
node_id = 1; %root node

%train binary split tree (BGOT without feature selection)
[ hier ] = append_constructRecursiveNode( features, featurei, class_id, traj', fold, classifier, traj_vote, class_set, node_id);
hier.b_isTree = 1;
hier.node_classSet = class_set;
%an example of using forward sequential feature selection to select feature
%subset for BGOT. Note: if this function does not work for you, you may try
%to do feature selection first and assign Subfeature by using
%append_hier_assignFSSubset. 
%
%append_hierFeatureSelection = feature selection + append_hier_assignFSSubset
[ BGOT ] = append_hierFeatureSelection( hier, features, featurei, class_id, traj', '.');
%set Root Subfeature field for node rejection
[featureSubset, scoreResult] = classify_featureselection_fw(features, featurei, class_id, traj', length(unique(featurei)), 3, fold, classifier, 'fs_root');
[mv, mi] = max(scoreResult(:, standard));
BGOT.Root.Subfeature = featureSubset(1:mi);

%use cross validation for evaluation
%[  result] = classify_crossValidation(features, featurei, class_id, traj', fold, classifier, 0, BGOT, 1);
end