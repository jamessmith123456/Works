function texture = feature_getCoOccurrenceMatrix(rgbImg,maskImg)

rgbImg = append_getSpecificImage(rgbImg,maskImg);

stepD = 1;
maxD = 10;
dd = 1;
NL = 8;

for d = stepD:stepD:maxD
      d2=round(d/sqrt(2));
      glcm = append_imscomatrix(rgbImg, 'numlevels',NL,'offset', [0 d; -d2 d2; -d 0; -d2 -d2], 'Symmetric', true, 'Invariant', true);
      % get the texture measures excluding the black pixels (graylevel=0)
      stats = append_fastcoprops(glcm(2:end,2:end,:));
   
      stats.Contrast = (stats.Contrast-mean(stats.Contrast))/std(stats.Contrast);
      stats.Correlation = (stats.Correlation-mean(stats.Correlation))/std(stats.Correlation);
      stats.Energy = (stats.Energy-mean(stats.Energy))/std(stats.Energy);
      stats.Entropy = (stats.Entropy-mean(stats.Entropy))/std(stats.Entropy);
      stats.Homogeneity = (stats.Homogeneity-mean(stats.Homogeneity))/std(stats.Homogeneity);
      stats.InvDiffMoment = (stats.InvDiffMoment-mean(stats.InvDiffMoment))/std(stats.InvDiffMoment);
      stats.ClusterShade = (stats.ClusterShade-mean(stats.ClusterShade))/std(stats.ClusterShade);
      stats.ClusterProminence = (stats.ClusterProminence-mean(stats.ClusterProminence))/std(stats.ClusterProminence);
      stats.MaxProbability = (stats.MaxProbability-mean(stats.MaxProbability))/std(stats.MaxProbability);
      stats.Autocorrelation = (stats.Autocorrelation-mean(stats.Autocorrelation))/std(stats.Autocorrelation);
      stats.Dissimilarity = (stats.Dissimilarity-mean(stats.Dissimilarity))/std(stats.Dissimilarity);
      stats.Variance = (stats.Variance-mean(stats.Variance))/std(stats.Variance);
      
      texture(1:72,dd)=[stats.Contrast,stats.Correlation,stats.Energy,stats.Entropy,...
          stats.Homogeneity,stats.InvDiffMoment,stats.ClusterShade,stats.ClusterProminence,...
          stats.MaxProbability,stats.Autocorrelation,stats.Dissimilarity,stats.Variance];
      dd=dd+1;
end

end