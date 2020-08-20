function pp = feature_anna_phogDescriptor(bh,bv,L,bin)
% anna_PHOGDESCRIPTOR Computes Pyramid Histogram of Oriented Gradient over a ROI.
%               
%IN:
%	bh - matrix of bin histogram values
%	bv - matrix of gradient values 
%   L - number of pyramid levels
%   bin - number of bins
%
%OUT:
%	p - pyramid histogram of oriented gradients (phog descriptor)

p = [];
%level 0
for b=1:bin
    ind = bh==b;
    p = [p;sum(bv(ind))];
end

pp = cell(L+1,1);
pp{1} = p;
        
cella = 1;
for l=1:L
    p = [];
    x2 = size(bh,2)/(2^l);
    y2 = size(bh,1)/(2^l);
    x = fix(x2);
    y = fix(y2);
    xx2=0;
    yy2=0;
    while xx2+x2<=size(bh,2)
        xx = fix(xx2);
        while yy2 +y2 <=size(bh,1) 
            yy=fix(yy2);
            bh_cella = [];
            bv_cella = [];
            
            bh_cella = bh(yy+1:yy+y,xx+1:xx+x);
            bv_cella = bv(yy+1:yy+y,xx+1:xx+x);
            
            for b=1:bin
                ind = bh_cella==b;
                p = [p;sum(bv_cella(ind))];
            end 
            yy2 = yy2+y2;
        end        
        cella = cella+1;
        yy2 = 0;
        xx2 = xx2+x2;
    end
    pp{l+1}=p;
end

for l = 1:L+1
    s = sum(pp{l});
    if s ~=0
        pp{l} = pp{l} / s;
    end
end
% if sum(p)~=0
%     p = p*(L+1)/sum(p);
% end
end
