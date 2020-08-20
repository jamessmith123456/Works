function hist = feature_bog(descrs, vocab) %, frames, im)

% --------------------------------------------------------------------
% Xuan (Phoenix) Huang                             Xuan.Huang@ed.ac.uk
% University of Edinburgh                                    Sep. 2011
% Modified from VLFeat sourcecode_calth101
% --------------------------------------------------------------------

% --------------------------------------------------------------------
%                                                        Configuration
% --------------------------------------------------------------------
quantizer = 'kdtree' ;

% --------------------------------------------------------------------
%                                           Compute spatial histograms
% --------------------------------------------------------------------

% get PHOW features
%[frames, descrs] = vl_phow(im, model.phowOpts{:}) ;

% quantize appearance
switch quantizer
  case 'vq'
    [drop, binsa] = min(vl_alldist(vocab, single(descrs)), [], 1) ;
  case 'kdtree'
    kdtree = vl_kdtreebuild(vocab) ;
    binsa = double(vl_kdtreequery(kdtree, vocab, ...
                                  single(descrs), ...
                                  'MaxComparisons', 15)) ;
end

numWords = size(vocab, 2) ;

hist = zeros(numWords, 1) ;
hist = vl_binsum(hist, ones(size(binsa)), binsa);

hist = single(hist / sum(hist)) ;

%featureSpatial = true;
% if featureSpatial
%     width = size(im,2) ;
%     height = size(im,1) ;
%     
%     numSpatialX = 2 ;
%     numSpatialY = 2 ;
%     
%     for i = 1:length(numSpatialX)
%         binsx = vl_binsearch(linspace(1,width,numSpatialX(i)+1), frames(1,:)) ;
%         binsy = vl_binsearch(linspace(1,height,numSpatialY(i)+1), frames(2,:)) ;
%         
%         % combined quantization
%         bins = sub2ind([numSpatialY(i), numSpatialX(i), numWords], ...
%             binsy,binsx,binsa) ;
%         hist = zeros(numSpatialY(i) * numSpatialX(i) * numWords, 1) ;
%         hist = vl_binsum(hist, ones(size(bins)), bins) ;
%         hists{i} = single(hist / sum(hist)) ;
%     end
% end
% hist = cat(1,hists{:}) ;
% hist = hist / sum(hist) ;

end
