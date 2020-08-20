function [OutContour] = interp_contour(Contour, nr_of_points, part)
%
% Ing Dr B.J.Boom                      bboom@inf.ed.ac.uk
% Edinburgh, United Kingdom                    March 2011 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%
if nargin < 3
    part = 0;
end
    
if ~part
    Eucl = sqrt(sum((Contour(1:end,:) - Contour([2:end,1],:)).^2,2));
else
    Eucl = sqrt(sum((Contour(1:end-1,:) - Contour(2:end,:)).^2,2));
end
sumEucl = sum(Eucl);
step_dist = sumEucl/nr_of_points;
cum_Eucl = [0; cumsum(Eucl)];

city_dis = (Contour([2:end,1],:) - Contour([1:end-1,1],:));
warning off MATLAB:divideByZero
div = city_dis(:,2)./city_dis(:,1);
alpha = atan(city_dis(:,2)./city_dis(:,1));
warning on MATLAB:divideByZero

OutContour = zeros(nr_of_points,2);
OutContour(1,:) = Contour(1,:);
dist = 0;

for iter=2:nr_of_points
    dist = dist + step_dist;
    mask = cum_Eucl <= dist;
    [max_value, max_index] = max(cum_Eucl(mask));
    OutContour(iter,1) = Contour(max_index,1) + sign(city_dis(max_index,1))  * ((dist-max_value) * cos(alpha(max_index)));
    OutContour(iter,2) = Contour(max_index,2) + sign(city_dis(max_index,1)) * ((dist-max_value) * sin(alpha(max_index)));   
end



