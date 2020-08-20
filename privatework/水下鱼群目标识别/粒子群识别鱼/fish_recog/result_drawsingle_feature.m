function result_drawsingle_feature( feature )
%RESULT_DRAWSINGLE_FEATURE Summary of this function goes here
%   Detailed explanation goes here
showformat = ['b+'; 'g+'; 'r+'; 'k+'; 'c+'; 'm+'; 'bs'; 'gs'; 'rs'; 'ks'; 'cs'; 'ms'];

[m,n] = size(feature);
figure(1);

for j = 1:n
hold on;

for i = 1:m
    colorindex = 1+int8(floor((i-1) / 45));
    plot(i, feature(i,j), showformat(colorindex,:));
end

hold off;
pause();
close(1);
end

end

