function trainpsix = feature_kerneltrans(trainfeature)
% --------------------------------------------------------------------
%                                                  Compute feature map
% --------------------------------------------------------------------
trainpsix = vl_homkermap(trainfeature', 1, 'kchi2', 'gamma', .5) ;

end

