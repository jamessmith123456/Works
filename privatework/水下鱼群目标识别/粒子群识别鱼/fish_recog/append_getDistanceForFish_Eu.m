% Returns the score of a fish [speciesID, diatance]

% Fish1: Fish to be classified
% Fish2: Fish comparing against
% SpeciesId: Species of compared fish
% festureVector: vector of feature indexs to use

function score = append_getDistanceForFish_Eu(feature1, feature2, featureindex)
    score = 0;
    for i=1:length(featureindex)
    	score = score + append_getFeatureDistances(feature1,feature2,featureindex(i));
    end
end