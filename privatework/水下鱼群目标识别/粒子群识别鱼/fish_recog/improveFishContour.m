function [Image,Mask2] = improveFishContour(Image,Mask,History,Valid)

%
% ing Dr B.J.Boom                      bboom@inf.ed.ac.uk
% Edinburgh, United Kingdom                    March 2011 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%

rows = 1:size(Image,1);
cols = 1:size(Image,2);
 
rows = rows(sum(Valid,2)>0);
cols = cols(sum(Valid,1)>0);

im = Image(rows,cols,:);
temp_mask = Mask(rows,cols); 

Diff = abs(double(Image)/255 - History/255);
Diff = max(Diff,[],3);

%figure(2); imagesc(Diff);
%figure(1);

mask1 = zeros(size(temp_mask));
%mask1(logical(temp_mask)) = 1;
mask1(Diff(rows,cols) > 0.2 | logical(temp_mask)) = 1;
mask1(1,1:end) = -1;
mask1(end,1:end) = -1;
mask1(1:end,1) = -1;
mask1(1:end,end) = -1;

warning off;
[L, dc, sc, vC, hC] = GrabCut(double(im)/255, mask1);
warning on;

Mask2 = zeros(size(Mask));
Mask2(rows,cols) = L > 0;

La = bwlabel(logical(Mask2));

uniq_l = setdiff(unique(La),0)';
num_of_val = zeros(size(uniq_l));
for ii = uniq_l
    num_of_val(ii) = sum(La(:) == ii);
end

[max_v max_i] = max(num_of_val);
Mask2 = La == max_i;

end



