% Create a ReconstructionICA object by using the rica function.
% ����caltech101����patches.
data = load('caltech101patches');
size(data.X)
% һ��10000��patches,ÿ��patch����363��������������ȡ100ά����

q = 100;
Mdl = rica(data.X,q,'IterationLimit',100)

newfeature = data.X*Mdl.TransformWeights;

