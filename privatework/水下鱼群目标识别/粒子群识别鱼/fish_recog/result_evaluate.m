function [result] = result_evaluate(predictClass, classid, removeMinus)

% --------------------------------------------------------------------
% Xuan (Phoenix) Huang                             Xuan.Huang@ed.ac.uk
% University of Edinburgh                                    Sep. 2011
% --------------------------------------------------------------------

if ~exist('removeMinus','var'), removeMinus = 0;end

if isempty(predictClass) || isempty(classid)
    result.inValid = 1;
    return;
end

confus = confusionmat(classid, predictClass);

result.confus = confus;
result.recallCount = (sum(diag(confus))/sum(sum(confus)));

true_posive = diag(confus);
all_positive = max(0, sum(confus,1)');
groundtruth = max(0, sum(confus,2));

result.classnum = groundtruth';
result.returnnum = all_positive';
result.correct = true_posive';

result.classrecall = (true_posive./(eps+groundtruth))';
result.classrecall(isnan(result.classrecall))=0;
if removeMinus
    result.classrecall = result.classrecall(2:end);
end
result.recallAver = mean(result.classrecall);

result.classprecision = (true_posive./(eps+all_positive))';
result.classprecision(isnan(result.classprecision))=0;
if removeMinus
    result.classprecision = result.classprecision(2:end);
end
result.precisionAver = mean(result.classprecision);
end