function temphist = feature_getFourierDescriptors(binImg, fourier_size)
    K_cout = feature_curvecorner(binImg);
    if isempty(K_cout)
        K_cout = [0,0];
    end
    FDescript = fft(K_cout(:,1)+K_cout(:,2)*1i);
    %FDescript = (FDescript-mean(FDescript))/std(FDescript);
    
    fourier_number = min(fourier_size, length(FDescript));
    temphist = zeros(fourier_size,1);
    for i=1:fourier_number
        temphist(i) = norm(FDescript(i));
    end

    temphist = temphist / sum(abs(temphist));
end