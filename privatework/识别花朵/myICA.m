classdef myICA < nnet.layer.Layer   
    methods
        function layer = myICA()
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = 'myICA';
            % Layer constructor function goes here.
        end
        
        function [Z] = predict(~, X)
            % Forward input data through the layer at prediction time and
%             Z = X * myrica(X,1000,'IterationLimit',100);
%             [m,n,k] = size(X);%m:1 n:1 k:4096
%             XX = X;
%             temp = myrica(XX,1000,'IterationLimit',100);
%             X(:,:,k-100:k)=0;
%             Z = X;
%             Z = max(0, X) + 0.1.* min(0, X);
            Z = myrica2(X);
%             Z = max(0, X) + 0.1.* min(0, X);
%             ZZ = reshape([X(:,:,1:100),zeros(1,m,n-100)],1,m,n);
%             temp = myrica(X,1000,'IterationLimit',100);
%             disp(['real:',num2str(isreal(X))]);
%             disp(['matrix:',num2str(ismatrix(X))]);
%             disp(['numeric:',num2str(isnumeric(X))]);
%             Z = reshape([X(:,1:100),zeros(m,n-100)],1,m,n);
%             Z = reshape([temp,zeros(m,n-1000)],1,m,n);
        end

        function [dLdX] = backward(~, ~, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            dLdX = 1.*dLdZ;
            % Layer backward function goes here.
        end
    end
end