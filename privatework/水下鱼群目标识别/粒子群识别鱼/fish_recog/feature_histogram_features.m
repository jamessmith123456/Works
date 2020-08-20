function feature = feature_histogram_features( Histogram )
% ------------
% Description:
% ------------
%  This function is to obtain state of the art histogram based features
% such as:
%   Mean
%   Variance
%   Skewness
%   Kurtosis
%   Energy
%   Entropy
% ---------
% History:
% ---------
% Creation: beta         Date: 09/11/2007
%----------
% Example:
%----------
% Stats = append_histogram_features( I,'NumLevels',9,'G',[] )
%
% -----------
% Author:
% -----------
%    (C)Xunkai Wei <xunkai.wei@gmail.com>
%    Beijing Aeronautical Technology Research Center
%    Beijing %9203-12,10076
%
[samle_size, drop] = size(Histogram);
for i = 1:samle_size
    feature(i,:)=inner_single_histogram(Histogram(i,:));
end
end
function stats = inner_single_histogram(Histogram)
Histogram = Histogram - min(Histogram);
Gray_vector = 1:length(Histogram);
%--------------------------------------------------------------------------
% 2. Now calculate its histogram statistics
%--------------------------------------------------------------------------
% Calculate obtains the approximate probability density of occurrence of the intensity
% levels
Prob                = Histogram./(sum(Histogram));
% 2.1 Mean 
Mean                = sum(Prob.*Gray_vector);
% 2.2 Variance
Variance            = sum(Prob.*(Gray_vector-Mean).^2);
% 2.3 Skewness
Skewness            = calculateSkewness(Gray_vector,Prob,Mean,Variance);
% 2.4 Kurtosis
Kurtosis            = calculateKurtosis(Gray_vector,Prob,Mean,Variance);
% 2.5 Energy
Energy              = sum(Prob.*Prob);
% 2.6 Entropy
ProbPos = Prob(Prob > 0);
Entropy             = -sum(ProbPos.*log(ProbPos));
%-------------------------------------------------------------------------
% 3. Insert all features and return
%--------------------------------------------------------------------------
stats =[Mean Variance Skewness Kurtosis  Energy  Entropy];
stats(isnan(stats)) = 0;
end
% End of funtion
%--------------------------------------------------------------------------
% Utility functions
%--------------------------------------------------------------------------
function Skewness = calculateSkewness(Gray_vector,Prob,Mean,Variance)
% Calculate Skewness
term1    = Prob.*(Gray_vector-Mean).^3;
term2    = sqrt(Variance);
Skewness = term2^(-3)*sum(term1);
end

function Kurtosis = calculateKurtosis(Gray_vector,Prob,Mean,Variance)
% Calculate Kurtosis
term1    = Prob.*(Gray_vector-Mean).^4;
term2    = sqrt(Variance);
Kurtosis = term2^(-4)*sum(term1);
end