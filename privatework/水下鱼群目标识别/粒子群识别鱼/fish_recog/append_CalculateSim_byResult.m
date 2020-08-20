function similarity = append_CalculateSim_byResult(feature, classid)

[samples, dimens] = size(feature);
similarity = zeros(dimens, dimens);
for i = 1:dimens
    for j = i:dimens
        similarity(i, j) = 1;
        similarity(j, i) = similarity(i, j);
    end
end

end

