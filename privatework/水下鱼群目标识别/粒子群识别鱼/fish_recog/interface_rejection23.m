function [ prediction, score_single, scores ] = interface_rejection23( prediction, score_single, scores, features, model_data )

for i = 1:length(model_data.rejectClass)
    rejection_indic = find(prediction == model_data.rejectClass(i));
    if ~isempty(model_data.model{i}.indic) && ~isempty(rejection_indic)
        tmp_score=classify_GMM_predict(features(rejection_indic,:), model_data.model{i});
        
        tmp_score=log(eps+tmp_score)-log(eps);
        tmp_score = tmp_score / model_data.maxScore(i);
        tmp_score(tmp_score>1)=1;tmp_score(tmp_score<0)=0;
        
        score_single(rejection_indic) = tmp_score;

        rejection_reject = find(tmp_score<model_data.rejectThres(i));
        scores(rejection_indic(rejection_reject),:) = 0;
        prediction(rejection_indic(rejection_reject)) = 0;
    end
    
end

end


