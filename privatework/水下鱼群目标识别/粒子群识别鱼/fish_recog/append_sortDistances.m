function distances = append_sortDistances(distances)

[t N] = size(distances);

tmp = zeros(2,1);
    for i = 1 : N-1
        for j = i+1 : N
            if distances(1,i) > distances(1,j)
                tmp(1,1) = distances(1,i);
                tmp(2,1) = distances(2,i);
                distances(1,i) = distances(1,j);
                distances(2,i) = distances(2,j);
                distances(1,j) = tmp(1,1);
                distances(2,j) = tmp(2,1);
            end
        end
    end
end