% Create a ReconstructionICA object by using the rica function.
% 导入caltech101数据patches.
data = load('caltech101patches');
size(data.X)
% 一共10000个patches,每个patch包含363个特征，现在提取100维特征

q = 100;
Mdl = rica(data.X,q,'IterationLimit',100)

newfeature = data.X*Mdl.TransformWeights;

