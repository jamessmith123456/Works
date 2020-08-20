function [frames descrs] = feature_dsift(rgbimg, maskimg,varargin)

% --------------------------------------------------------------------
% Xuan (Phoenix) Huang                             Xuan.Huang@ed.ac.uk
% University of Edinburgh                                    Sep. 2011
% Modified from VLFeat sourcecode_calth101
% --------------------------------------------------------------------

% --------------------------------------------------------------------
%                                                        Configuration
% --------------------------------------------------------------------
phowOpts = {'Verbose', false, 'Sizes', 7, 'Step', 5, 'Color', 'rgb'} ;
phowOpts = {phowOpts{:} , varargin{:}} ;

im = im2single(rgbimg) ;

%rgbimg = standarizeImage(rgbimg) ;
[frames, descrs] = vl_phow(im, phowOpts{:}) ;

vecinside = [];
for ivalid = 1:length(frames)
    x = int32(frames(2, ivalid)); y = int32(frames(1, ivalid));
    if (maskimg(x, y) > 128)
        vecinside(end+1) = ivalid;
    end
end
%hold off;

frames = frames(:, vecinside);
descrs = descrs(:, vecinside);