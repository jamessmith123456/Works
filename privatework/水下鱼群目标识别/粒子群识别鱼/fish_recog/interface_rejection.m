function [ prediction, score_single, scores ] = interface_rejection( prediction, score_single, scores, features, model_data )

rejection_class = 1:6;
rejection_thres = 0.01;

prediction_15 = model_data.convert2database(prediction, 3);
rejection_indic = find(ismember(prediction_15, rejection_class));
[ post_log_score ] = classify_rejectionGMM( model_data.GMM_models, features(rejection_indic, :), prediction_15(rejection_indic), model_data.GMM_prior );
rejection_reject = find(post_log_score < rejection_thres);

score_single(rejection_indic) = post_log_score;
scores(rejection_indic(rejection_reject),:) = 0;
prediction(rejection_indic(rejection_reject)) = 0;
end

