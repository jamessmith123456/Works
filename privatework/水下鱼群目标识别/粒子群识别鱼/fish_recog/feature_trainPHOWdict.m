function vocab = feature_trainPHOWdict(trainimageset, maskimageset, VocabSet, numWords)

% --------------------------------------------------------------------
% Xuan (Phoenix) Huang                             Xuan.Huang@ed.ac.uk
% University of Edinburgh                                    Sep. 2011
% Modified from VLFeat sourcecode_calth101
% --------------------------------------------------------------------

% --------------------------------------------------------------------
%                                                        Configuration
% --------------------------------------------------------------------
% --------------------------------------------------------------------
%                                                     Train vocabulary
% --------------------------------------------------------------------

phowOpts = {'Sizes', 7, 'Step', 5, 'Color', 'rgb'} ;

% Get some PHOW descriptors to train the dictionary
trainsamplenumber = length(trainimageset);
selTrainFeats = vl_colsubset(1:trainsamplenumber, VocabSet) ;

descrs = cell(length(selTrainFeats), 1);
for ii = 1:length(selTrainFeats)
    index = selTrainFeats(ii);
    trainimg = cell2mat(trainimageset(index));
    maskimg = cell2mat(maskimageset(index));
    [drop, descrs{ii}] =  feature_dsift(trainimg, maskimg, phowOpts{:}) ;
end

descrs = vl_colsubset(cat(2, descrs{:}), 7e4) ;
descrs = single(descrs) ;

% Quantize the descriptors to get the visual words
vocab = vl_kmeans(descrs, numWords, 'algorithm', 'elkan') ;

end