classdef myICA < nnet.layer.Layer   
    methods
        function layer = myICA()
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = 'myICA';
            % Layer constructor function goes here.
        end
        
%         function [Z] = predict(~, X)
%             % Forward input data through the layer at prediction time and
%             Z = X * myrica(X,1000,'IterationLimit',100);
% %             Z = max(0, X) + 0.1.* min(0, X);
%         end

        function [Z] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %         memory      - Memory value for backward propagation

            % Layer forward function for training goes here.
            Z = X * myrica(X,1000,'IterationLimit',100);
        end

        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
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