% Returns the score of a fish [speciesID, diatance]

% Fish1: Fish to be classified
% Fish2: Fish comparing against
% SpeciesId: Species of compared fish
% festureVector: vector of feature indexs to use

function score = append_getDistanceForFish_Mh(feature1, feature2, featureindex)
    feature2 = feature2(:,featureindex);
    feature1 = feature1(:,featureindex);
    
    score = mahal(feature1,feature2);
end