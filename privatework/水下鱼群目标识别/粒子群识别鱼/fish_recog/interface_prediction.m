function [GMM_result_predict, GMM_result_score, traj_predict_convert, traj_scores, traj_array ] = interface_prediction( rgbImg, binImg, model_data, traj_id )

%feature
[features_raw]= interface_generateFeatureSet(rgbImg, binImg);

%BGOT
[features_BGOT]= interface_normalizeFeatureFromPrefix(features_raw, model_data.Feature_model.feature_mean, model_data.Feature_model.feature_std);
[ BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23 ] = interface_classification( features_BGOT, model_data.BGOT_1301_69f );

%image reject
[image_accpet] = append_acceptImage(rgbImg, binImg);
BGOT_result_predict(image_accpet==0,:) = 0;
BGOT_result_score(image_accpet==0,:) = 0;
BGOT_result_scores_23(image_accpet==0,:) = 0;

%GMM reject
[features_GMM]= interface_normalizeFeatureFromPrefix(features_raw, model_data.Feature_model.prefix95_mean, model_data.Feature_model.prefix95_std);
[GMM_result_predict, GMM_result_score,  GMM_result_score_23] = interface_rejection23( BGOT_result_predict, BGOT_result_score, BGOT_result_scores_23, features_GMM, model_data.GMM_model );

%trajectory voting
[traj_predict, traj_scores, traj_array] = result_trajvote_byScore( GMM_result_score_23, traj_id);

%result convert
GMM_result_predict(GMM_result_predict>0) = model_data.convert2database(GMM_result_predict(GMM_result_predict>0), 2);
traj_predict_convert = result_convertClassCell(traj_predict, model_data.convert2database(:, 2));

end