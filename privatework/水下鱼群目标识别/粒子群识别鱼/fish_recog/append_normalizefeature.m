function [ nfeature ] = append_normalizefeature( feature )

sam = size(feature,1);
dim = size(feature,2);

nfeature = zeros(sam, dim);
for i = 1:dim
    fea = feature(:,i);
    nfeature(:,i) = (fea - mean(fea)) / max(0.00001,std(fea));
end

end

