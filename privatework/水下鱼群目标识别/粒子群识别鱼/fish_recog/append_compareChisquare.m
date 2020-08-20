function val = append_compareChisquare(hist1,hist2)

val = 0;
iter = size(hist1,2);
for j=1:iter
   if  ~(hist1(j) == 0.0 && hist2(j) == 0.0)
       temp=hist1(j)-hist2(j);
       val = val + ((temp*temp)/(hist1(j)+hist2(j))) ;
   end
end
