%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function defines ckassifier used. This is a K nearest neighbour
% classifier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TestFish,
% TrainFish
% featuresOne,
% featuresTwo,
% featuresThree,
% featuresFour
% knn

function result = classify_KNN(feature_train, classid_train, feature_test, classid_test, knn, featureindex)
    trainfishnum = length(classid_train);
    testfishnum = length(classid_test);    
    classnum = max(classid_test);

    confus = zeros(classnum, classnum);
    predictclass = zeros(testfishnum, 1);
    
    for k=1:testfishnum %loop through all test fish
        feature1 = feature_test(k, :);
        
        distances = zeros(2, classnum);
        
        for m = 1:classnum
            trainid = classid_train == m;
            distances(1,m) = append_getDistanceForFish_Mh(feature1,feature_train(trainid,:), featureindex);
            distances(2,m) = m;            
        end
        
        % Sort distances according to lowest size.
        results = append_sortDistances(distances);

        % Determine species choice
        trueid = classid_test(k);
        predict = append_getMedianKNN(knn,results);
        
        predictclass(k) = predict;
        confus(trueid, predict) = confus(trueid, predict) + 1;
    end
    
    result.predict = predictclass;
    result.precision = (sum(diag(confus))/sum(sum(confus)));
    pt = diag(confus);
    tt = sum(confus,2);

    classaverageset = pt./tt;
    classaverageset = classaverageset(tt ~= 0);
    result.classaverageset = classaverageset;
    result.avarageP = mean(classaverageset);

    result.confus = confus;
    result.classprecision = (pt./tt)';
end