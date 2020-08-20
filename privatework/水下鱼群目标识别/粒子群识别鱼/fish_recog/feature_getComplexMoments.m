function [ci1 ci2 ci3 ci4 ci5 ci6] = feature_getComplexMoments(binImg)

    area = bwarea(binImg);
    for i=1:size(binImg,1)
        for j=1:size(binImg,2)
            if(binImg(i,j) == 255)
                binImg(i,j) = 1;
            end
        end
    end
%Get the specific scale invariabt moments
    s11 = append_complexmoment(binImg,1,1) / (area^2);
    s20 = append_complexmoment(binImg,2,0) / (area^2);
    s21 = append_complexmoment(binImg,2,1) / (area^2.5);
    s12 = append_complexmoment(binImg,1,2) / (area^2.5);
    s30 = append_complexmoment(binImg,3,0) / (area^2.5);
 % make rotation invariants    
    ci1 = real(s11);
    ci2 = real(1000*s21*s12);
    ci3 = 10000*real(s20*s12*s12);
    ci4 = 10000*imag(s20*s12*s12);
    ci5 = 1000000*real( s30*s12*s12*s12);
    ci6 = 1000000*imag( s30*s12*s12*s12);
end