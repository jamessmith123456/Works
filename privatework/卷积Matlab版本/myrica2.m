function [result] = myrica2(X)
%     result = max(0,X)+0.1.*min(0,X);
    [coeff, score, latent, tsquared, explained] = pca(X);
%     p=1000;
    result = score;
%     result = [score(:,1:p),X(:,p+1:end)];
%     [m,n] = size(X);
%     result = reshape(result,1,m,n);
%     result = myrica(X,1000,'IterationLimit',100);
end