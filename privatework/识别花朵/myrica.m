function [result] = myrica(X,Q,varargin)
%rica - Create a reconstruction ICA (RICA) object.
%   OBJ = rica(X,Q) takes a N-by-P predictor matrix X and an integer Q and
%   applies reconstruction ICA (RICA) to learn a P-by-Q matrix of
%   transformation weights. N is the number of observations, P is the
%   number of predictors and Q is the desired number of new features. OBJ
%   is an object of type ReconstructionICA.
%
%   OBJ = rica(TBL,Q) is an alternative syntax that accepts a table TBL.
%   All variables in TBL must be numeric variables.
%
%   o The learned transformation weights can be accessed using the
%   TransformWeights property of OBJ.
%
%   o A N-by-P matrix X of original features can be transformed into a
%   N-by-Q matrix of new features Z via the TRANSFORM method of OBJ.
%
%   OBJ = rica(X,Q,varargin) also accepts the following additional
%   name/value pairs to control the fitting:
% 
%       'IterationLimit'  - A positive integer specifying the maximum 
%                           number of iterations. Default is 1000.
%       'VerbosityLevel'  - A non-negative integer specifying the verbosity
%                           level as follows:
%                           * 0   - no convergence summary is displayed.
%                           * >=1 - convergence summary is displayed on
%                                   screen.
%                           Default is 0.
%       'Lambda'          - A non-negative real scalar specifying the
%                           penalty coefficient for the reconstruction term
%                           in the objective function. Default is 1.
%       'Standardize'       Logical scalar. If true, standardize predictors 
%                           by centering and dividing columns by their 
%                           standard deviations. Default is false.
%       'ContrastFcn'       A character vector specifying the contrast
%                           function for RICA. Choices are 'logcosh', 'exp'
%                           or 'sqrt'. Default is 'logcosh'.
%       'InitialTransformWeights'
%                         - A P-by-Q matrix of transformation weights to 
%                           initialize the optimization for transformation
%                           weights. Default is [] indicating that initial
%                           transformation weights should be chosen
%                           randomly.
%       'NonGaussianityIndicator'
%                         - A length Q vector sgn with elements +1 or -1
%                           specifying the type of non-Gaussianity of the
%                           sources. If sgn(k) = +1 then k-th source is
%                           assumed to be super-Gaussian (sharp peak at
%                           zero). If sgn(k) = -1 then k-th source is
%                           assumed to be sub-Gaussian. Default is
%                           ones(Q,1).
%       'GradientTolerance' 
%                         - A positive real scalar specifying the relative
%                           convergence tolerance on the norm of the
%                           objective function gradient. Default is 1e-6.
%       'StepTolerance'   - A positive real scalar specifying the absolute
%                           convergence tolerance on the step size. Default
%                           is 1e-6.
%
%   Example: Extract features from Caltech101 image patches.
%       % 1. Load Caltech101 image patches.
%       data = load('caltech101patches');
%       data.Description
%       % 2. Extract Q new features.
%       Q = 100;
%       X = data.X(1:10000,:);
%       obj = rica(X,Q,'VerbosityLevel',1,'IterationLimit',1000,'NonGaussianityIndicator',ones(Q,1),'Lambda',1,'GradientTolerance',1e-3,'InitialTransformWeights',rand(363,Q));
%       % 3. Plot transformation weights as images.
%       W = obj.TransformWeights;
%       W = reshape(W,[11,11,3,Q]);
%       [dx,dy,~,~] = size(W);
%       for f = 1:Q
%           Wvec = W(:,:,:,f);
%           Wvec = Wvec(:);
%           Wvec =(Wvec - min(Wvec))/(max(Wvec) - min(Wvec));
%           W(:,:,:,f) = reshape(Wvec,dx,dy,3);
%       end
%       m   = ceil(sqrt(Q));
%       n   = m;
%       img = zeros(m*dx,n*dy,3);
%       f   = 1;
%       for i = 1:m
%           for j = 1:n
%               if (f <= Q)
%                   img((i-1)*dx+1:i*dx,(j-1)*dy+1:j*dy,:) = W(:,:,:,f);
%                   f = f+1;
%               end
%           end
%       end
%       imshow(img,'InitialMagnification',300);
%       % 4. Alternative way to plot weights (requires Image Processing Toolbox).
%       W = obj.TransformWeights;
%       W = reshape(W,[11,11,3,Q]);
%       for f = 1:Q
%           W(:,:,:,f) = mat2gray(W(:,:,:,f));
%       end
%       W = imresize(W,5);
%       montage(W);
%
%   See also ReconstructionICA, sparsefilt.

%   Copyright 2016 The MathWorks, Inc.

    obj = ReconstructionICA(X,Q,varargin{:});
    myTransformWeights = obj.TransformWeights;
    result = double(X * myTransformWeights);
end