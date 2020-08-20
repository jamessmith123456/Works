function [L, dc , sc, vC, hC] = GrabCut(im, initMap)
%
% Performs segmentation of image into foreground/background
% regions
%
% Usage:
%   L = GrabCut(im, initMap)
%
% Inputs:
%   im - color image to be segmented
%   initMap - initial labeling of which we are _certain_ of:
%             1  - FG
%             -1 - BG
%             0  - uncertain
%
% Output:
%   L - a segmentation (i.e., {0,1} labeling) of the give image.
%
% This implementation follows the paper
% Rother C., Kolmogorov V. and Blake A. "GrabCut"-Interactive
% Foreground Extraction using iterated Graph Cuts. SIGGRAPH 2004.
%

% constants:
Gamma = 50;
MaxItr = 100;
K = 3; % number of components in each mixture model
INF = 1000;


% working in Lab space
labim = im; % RGB2Lab(im);

[hC vC] = SmoothnessTerm(labim);
sc = Gamma.*[0 1; 1 0];

certain_fg = initMap == 1;
certain_bg = initMap == -1;

oL = initMap;

% how to label all the uncertain pixels?
if (sum(sum(abs(certain_fg))) == 0)
    oL(initMap==0) = 1;
else
    oL(initMap==0) = -1;
end

done_suc = false;

for itr=1:MaxItr,
    
    % GMM for foreground and background - global model
    logpFG = LocalColorModel(labim, K, oL==1); 
    logpBG = LocalColorModel(labim, K, oL==-1); 

    % force labeling of certain labeling
    logpBG(certain_fg) = INF;
    logpFG(certain_bg) = INF;
    
    dc = cat(3, logpBG, logpFG);

    gch = GraphCut('open', dc , sc, vC, hC);  
    gch = GraphCut('set', gch, int32(oL==1)); % initial guess - previous result
    [gch L] = GraphCut('expand', gch);
    L = (2*L-1); % convert {0,1} to {-1,1} labeling
    gch = GraphCut('close', gch);

    % stop if converged
    if sum(oL(:)~=L(:)) < .001*numel(L)
        done_suc = true;
        break;
    end
    oL = L;
end
if ~done_suc
    warning('GrabCut:GrabCut','Failed to converge after %d iterations', itr);
end

