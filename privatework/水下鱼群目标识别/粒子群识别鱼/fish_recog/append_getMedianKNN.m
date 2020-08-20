function class = append_getMedianKNN(k,results)

    class = results(2, 1);
    [M F] = mode(results(2, 1:k));
    
    decision = ceil(k/2);
    
    if F >= decision
        class = M;
    end
    
end