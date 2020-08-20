function [Rhist Ghist Bhist] = getSegmentedTailColourHistogram( Rhist, Ghist, Bhist)
    middle = size(Rhist,1)/2;
    for i=1:size(Rhist,1)
        if(i<middle)
            Rhist(i) = 0;
            Ghist(i) = 0;
            Bhist(i) = 0;
        end
    end
end