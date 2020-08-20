function [ hier ] = interface_constructBinaryHier(feature, featurei, classid, traj, fold, classifier, traj_vote, cv_one_fold)
%INTERFACE_CONSTRUCTHIER Summary of this function goes here
%   Detailed explanation goes here

if ~exist('cv_one_fold','var')
    cv_one_fold = 0;
end

hier = [];
if nargin < 4
    return;
end

if ~exist('classifier','var')
    classifier = 2;
end

if ~exist('fold','var')
    fold = 6;
end

%use trajctery result to vote prediction
if ~exist('traj_vote','var')
    traj_vote = 0;
end

node_id = 1;
[ hier ] = append_constructRecursiveNode( feature, featurei, classid, traj, fold, classifier, traj_vote, unique(classid), node_id, cv_one_fold);
hier.b_isTree = 1;
hier.Root = [];

end

