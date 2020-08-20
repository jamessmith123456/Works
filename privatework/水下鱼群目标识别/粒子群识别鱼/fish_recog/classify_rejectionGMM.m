function [ post_log_score, score_log_diag ] = classify_rejectionGMM( models, featureTest, classidPredict, prior )

[samples, dimens] = size(featureTest);
[sizeA, sizeB] = size(models);

scores_all = cell(sizeA, sizeB);
for i = 1:sizeA
for j = 1:sizeB
if ~isempty( models{i,j}.indic)
scores_all{i,j}=classify_GMM_predict(featureTest, models{i,j});
else
scores_all{i,j} = zeros(samples,1);
end
end
end

scores_log_total = cell(sizeA, 1);
scores_log_diag = zeros(sizeA, samples);
for i = 1:sizeA
tmp_score = [];
for j = 1:sizeB
tmp_score(j,:)=scores_all{i,j};
end
scores_log_total{i,1}=tmp_score;
scores_log_diag(i,:)=log(eps+scores_all{i,i})-log(eps);
end

post_log_scores = zeros(sizeA, samples);
for i = 1:sizeA
tmp_score = log(eps+scores_log_total{i,1})-log(eps);
for j = 1:samples
post_log_scores(i,j)=tmp_score(classidPredict(j),j)*prior(classidPredict(j))/(eps+sum(tmp_score(:,j).*prior));
end
end

post_log_score = zeros(samples, 1);
score_log_diag = zeros(samples, 1);
for i = 1:samples
    post_log_score(i,1)=post_log_scores(classidPredict(i),i);
    score_log_diag(i,1)=scores_log_diag(classidPredict(i),i);
end

end

