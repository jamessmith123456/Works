function [GLCMS] = append_imscomatrix(varargin)
%MSMATRIX Create a number of multispecral (rotation invariant) co-occurrence matrices, 
% generalization of GRAYCOMATRIX/COLOCOMATRIX to MultiSpectral images
% and hopefully depth images (RGB+Z)
% Lucia IPAB 090211
%
% input:  I is an image of size (N x M x K)
% 
% output: IGLCMS are all the interplane combinations of cooc matrices 
%
% es: for K=3 (RGB) 6 combination (RR,RG,RB,GG,GB,BB) for any of the other 
% parameters 
% (4 offset -> 4x6=24 GLCMS)
%         order: offset1 RR
%                offset2 RR
%                offset3 RR
%                offset4 RR
%                offset1 RG
%                offset2 RG
%                ...
%                offset4 BB
%
% option invariant=true ->
% GLCMS=mean(GLCMS) over offsets
%
% see comments of GRAYCOMATRIX for other input and output parameters


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GRAYCOMATRIX Create gray-level co-occurrence matrix.
%   GLCMS = GRAYCOMATRIX(I) analyzes pairs of horizontally adjacent pixels in
%   a scaled version of I.  If I is a binary image, it is scaled to 2
%   levels. If I is an intensity image, it is scaled to 8 levels. In this
%   case, there are 8 x 8 = 64 possible ordered combinations of values for
%   each pixel pair. GRAYCOMATRIX accumulates the total occurrence of each
%   such combination, producing a 8-by-8 output array, GLCMS. The row and
%   column subscripts in GLCMS correspond respectively to the first and second
%   (scaled) pixel-pair values.
%
%   GLCMS = GRAYCOMATRIX(I,PARAM1,VALUE1,PARAM2,VALUE2,...) returns one or more
%   gray-level co-occurrence matrices, depending on the values of the optional
%   parameter/value pairs. Parameter names can be abbreviated, and case does
%   not matter.
%   
%   Parameters include:
%  
%   'Offset'         A p-by-2 array of offsets specifying the distance between
%                    the pixel-of-interest and its neighbor. Each row in the
%                    array is a two-element vector, [ROW_OFFSET COL_OFFSET],
%                    that specifies the relationship, or 'Offset', between a
%                    pair of pixels. ROW_OFFSET is the number of rows between
%                    the pixel-of-interest and its neighbor.  COL_OFFSET is the
%                    number of columns between the pixel-of-interest and its
%                    neighbor. For example, if you want the number of
%                    occurrences where the pixel of interest is one pixel to the
%                    left of its neighbor, then [ROW_OFFSET COL_OFFSET] is 
%                    [0 1].
%  
%                    Because this offset is often expressed as an angle, the
%                    following table lists the offset values that specify common
%                    angles, given the pixel distance D.
%                            
%                    Angle     OFFSET
%                    -----     ------  
%                    0         [0 D]   
%                    45        [-D D]
%                    90        [-D 0]
%                    135       [-D -D]  
%  
%                    ROW_OFFSET and COL_OFFSET must be integers. 
%
%                    Default: [0 1]
%            
%   'NumLevels'      An integer specifying the number of gray levels to use when
%                    scaling the grayscale values in I. For example, if
%                    'NumLevels' is 8, GRAYCOMATRIX scales the values in I so
%                    they are integers between 1 and 8.  The number of gray levels
%                    determines the size of the gray-level co-occurrence matrix
%                    (GLCM).
%
%                    'NumLevels' must be an integer. 'NumLevels' must be 2 if I
%                    is logical.
%  
%                    Default: 8 for numeric
%                             2 for logical
%   
%   'GrayLimits'     A two-element vector, [LOW HIGH], that specifies how the
%                    grayscale values in I are linearly scaled into gray
%                    levels. Grayscale values less than or equal to LOW are
%                    scaled to 1. Grayscale values greater than or equal to
%                    HIGH are scaled to HIGH.  If 'GrayLimits' is set to [],
%                    GRAYCOMATRIX uses the minimum and maximum grayscale values
%                    in I as limits, [min(I(:)) max(I(:))].
%  
%                    Default: the LOW and HIGH values specified by the
%                    class, e.g., [LOW HIGH] is [0 1] if I is double and
%                    [-32768 32767] if I is int16.
%
%   'Symmetric'      A Boolean that creates a GLCM where the ordering of
%                    values in the pixel pairs is not considered. For
%                    example, when calculating the number of times the
%                    value 1 is adjacent to the value 2, GRAYCOMATRIX
%                    counts both 1,2 and 2,1 pairings, if 'Symmetric' is
%                    set to true. When 'Symmetric' is set to false,
%                    GRAYCOMATRIX only counts 1,2 or 2,1, depending on the
%                    value of 'offset'. The GLCM created in this way is
%                    symmetric across its diagonal, and is equivalent to
%                    the GLCM described by Haralick (1973).
%
%                    The GLCM produced by the following syntax, 
%
%                    graycomatrix(I, 'offset', [0 1], 'Symmetric', true)
%
%                    is equivalent to the sum of the two GLCMs produced by
%                    these statements.
%
%                    graycomatrix(I, 'offset', [0 1], 'Symmetric', false) 
%                    graycomatrix(I, 'offset', [0 -1], 'Symmetric', false) 
%
%                    Default: false
%  
%  
%   [GLCMS,SI] = GRAYCOMATRIX(...) returns the scaled image used to
%   calculate GLCM. The values in SI are between 1 and 'NumLevels'.
%
%   Class Support
%   -------------             
%   I can be numeric or logical.  I must be 2D, real, and nonsparse. SI is a
%   double matrix having the same size as I.  GLCMS is an
%   'NumLevels'-by-'NumLevels'-by-P double array where P is the number of
%   offsets in OFFSET.
%  
%   Notes
%   -----
%   Another name for a gray-level co-occurrence matrix is a gray-level
%   spatial dependence matrix.
%
%   GRAYCOMATRIX ignores pixels pairs if either of their values is NaN. It also
%   replaces Inf with the value 'NumLevels' and -Inf with the value 1.
%
%   GRAYCOMATRIX ignores border pixels, if the corresponding neighbors
%   defined by 'Offset' fall outside the image boundaries.
%
%   References
%   ----------
%   Haralick, R.M., K. Shanmugan, and I. Dinstein, "Textural Features for
%   Image Classification", IEEE Transactions on Systems, Man, and
%   Cybernetics, Vol. SMC-3, 1973, pp. 610-621.
%
%   Haralick, R.M., and L.G. Shapiro. Computer and Robot Vision: Vol. 1,
%   Addison-Wesley, 1992, p. 459.
%
%   Example 1
%   ---------      
%   Calculate the gray-level co-occurrence matrix (GLCM) and return
%   the scaled version of the image, SI, used by GRAYCOMATRIX to generate the
%   GLCM.
%
%        I = [1 1 5 6 8 8;2 3 5 7 0 2; 0 2 3 5 6 7];
%       [GLCMS,SI] = graycomatrix(I,'NumLevels',9,'G',[])
%     
%   Example 2
%   ---------  
%   Calculate the gray-level co-occurrence matrix for a grayscale image.
%  
%       I = imread('circuit.tif');
%       GLCMS = graycomatrix(I,'Offset',[2 0])
%
%   Example 3
%   ---------
%   Calculate gray-level co-occurrences matrices for a grayscale image
%   using four different offsets.
%  
%       I = imread('cell.tif');
%       offsets = [0 1;-1 1;-1 0;-1 -1];
%       [GLCMS,SI] = graycomatrix(I,'Of',offsets); 
%
%   Example 4
%   ---------  
%   Calculate the symmetric gray-level co-occurrence matrix (the Haralick
%   definition) for a grayscale image.
%  
%       I = imread('circuit.tif');
%       GLCMS = graycomatrix(I,'Offset',[2 0],'Symmetric', true)
%  
%   See also GRAYCOPROPS.
%  
%   Copyright 1993-2006 The MathWorks, Inc.
%   $Revision.1 $  $Date: 2006/05/24 03:31:11 $
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[I, Offset, NL, GL, makeSymmetric, makeInvariant] = ParseInputs(varargin{:});

% Scale I so that it contains integers between 1 and NL.
if GL(2) == GL(1)
  SI = ones(size(I));
else
  slope = (NL-1) / (GL(2) - GL(1));
  intercept = 1 - (slope*(GL(1)));
  SI = round(imlincomb(slope,I,intercept,'double'));
end

% Clip values if user had a value that is outside of the range, e.g., double
% image = [0 .5 2;0 1 1]; 2 is outside of [0,1]. The order of the following
% lines matters in the event that NL = 0.
SI(SI > NL) = NL;
SI(SI < 1) = 1;

numOffsets = size(Offset,1);

if NL ~= 0

  % Create vectors of row and column subscripts for every pixel and its neighbor.
  s = size(I);
  [r,c] = meshgrid(1:s(1),1:s(2));
  r = r(:);
  c = c(:);

  % check how many GLCMs need to allocate
  n=size(SI,3);% number color planes 
  m=0;
  for i=1:n 
    for j=i:n 
      for k = 1 : numOffsets
         m=m+1;
      end
    end
  end
  
  % allocate
  GLCMS = zeros(NL,NL,m); 
    
  % extend to Z and multispectral images
  m=1;
  for i=1:n 
      for j=i:n
          for k = 1 : numOffsets
              GLCMS(:,:,m) = computeGLCM(r,c,Offset(k,:),SI(:,:,i),SI(:,:,j),NL);
              m=m+1;
          end
      end
  end
  
  for s=1:size(GLCMS,3)
    if makeSymmetric 
        % Reflect glcm across the diagonal
        glcmTranspose = GLCMS(:,:,s).';
        GLCMS(:,:,s) = GLCMS(:,:,s) + glcmTranspose;
    end
  end

  nGLCMS=size(GLCMS,3);
  nIGLCMS=nGLCMS/numOffsets;

  % allocate
  IGLCMS = zeros(NL,NL,nIGLCMS);
  if makeInvariant
      for i=1:nIGLCMS
          % average along directions
          for k=1:numOffsets
              IGLCMS(:,:,i)=IGLCMS(:,:,i)+GLCMS(:,:,(i-1)*numOffsets+k);
          end
          IGLCMS(:,:,i)=round(IGLCMS(:,:,i)/numOffsets);
      end
      % replace GLCMs with IGLCMs (Invariant cooc matrices), and remove original GLCMs matrices 
      GLCMS=IGLCMS;
      GLCMS(:,:,nIGLCMS+1:end)=[];
      % GLCMS=GLCMS(:,:,1:nIGLCMS); % trim
  end
  
else
  GLCMS = zeros(0,0,numOffsets);
end

% %-----------------------------------------------------------------------------
% function oneGLCM = computeGLCM(r,c,offset,si1,si2,nl)
% % computes GLCM given one Offset
% 
%   r2 = r + offset(1);
%   c2 = c + offset(2);
%   
%   % Remove pixel and its neighbor if they have subscripts outside the image
%   % boundary.
%   s = size(si1);
%   bad = find(c2 < 1 | c2 > s(2) | r2 < 1 | r2 > s(1));
%   Index = [r c r2 c2];
%   Index(bad,:) = []; %#ok
%   
%   % Create vectors containing the values of r and c (v1 and v2 respectively) for
%   % each pixel pair.
%   v1 = si1(sub2ind(s,Index(:,1),Index(:,2)));
%   v2 = si2(sub2ind(s,Index(:,3),Index(:,4)));
%   
%   % Make sure that v1 and v2 are column vectors.
%   v1 = v1(:);
%   v2 = v2(:);
%   
%   % Remove pixel and its neighbor if their value is NaN.
%   bad = isnan(v1) | isnan(v2);
%   if any(bad)
%     wid = sprintf('Images:%s:scaledImageContainsNan',mfilename);
%     msg = 'GLCM does not count pixel pairs if either of their values is NaN.';
%     warning(wid,'%s', msg);
%   end
%   Ind = [v1 v2];
%   Ind(bad,:) = [];
%   
%   if isempty(Ind)
%     oneGLCM = zeros(nl);
%   else
%     % Tabulate the occurrences of pixel pairs having v1 and v2.
%     oneGLCM = accumarray(Ind, 1, [nl nl]);
%   end
%   

%-----------------------------------------------------------------------------
function [I, offset, nl, gl, sym, rotinv] = ParseInputs(varargin)

narginchk(1,9);

% Check I
I = varargin{1};
for i=1:size(I,3)
validateattributes(I(:,:,i),{'logical','numeric'},{'2d','real','nonsparse'}, ...
              mfilename,'I',1);
end
% Assign Defaults
offset = [0 1];
if islogical(I)
  nl = 2;
else
  nl = 8;
end
gl = getrangefromclass(I);
sym = false;
rotinv = false;

% Parse Input Arguments
if nargin ~= 1
 
  paramStrings = {'Offset','NumLevels','GrayLimits','Symmetric','Invariant'};
  
  for k = 2:2:nargin

    param = lower(varargin{k});
    inputStr = validatestring(param, paramStrings, mfilename, 'PARAM', k);
    idx = k + 1;  %Advance index to the VALUE portion of the input.
    if idx > nargin
      eid = sprintf('Images:%s:missingParameterValue', mfilename);
      msg = sprintf('Parameter ''%s'' must be followed by a value.', inputStr);
      error(eid,'%s', msg);        
    end
    
    switch (inputStr)
     
     case 'Offset'
      
      offset = varargin{idx};
      validateattributes(offset,{'logical','numeric'},...
                    {'2d','nonempty','integer','real'},...
                    mfilename, 'OFFSET', idx);
      if size(offset,2) ~= 2
        eid = sprintf('Images:%s:invalidOffsetSize',mfilename);
        msg = 'OFFSET must be an n x 2 array.';
        error(eid,'%s',msg);
      end
      offset = double(offset);

     case 'NumLevels'
      
      nl = varargin{idx};
      validateattributes(nl,{'logical','numeric'},...
                    {'real','integer','nonnegative','nonempty','nonsparse'},...
                    mfilename, 'NL', idx);
      if numel(nl) > 1
        eid = sprintf('Images:%s:invalidNumLevels',mfilename);
        msg = 'NL cannot contain more than one element.';
        error(eid,'%s',msg);
      elseif islogical(I) && nl ~= 2
        eid = sprintf('Images:%s:invalidNumLevelsForBinary',mfilename);
        msg = 'NL must be two for a binary image.';
        error(eid,'%s',msg);
      end
      nl = double(nl);      
     
     case 'GrayLimits'
      
      gl = varargin{idx};
      validateattributes(gl,{'logical','numeric'},{'vector','real'},...
                    mfilename, 'GL', idx);
      if isempty(gl)
        gl = [min(I(:)) max(I(:))];
      elseif numel(gl) ~= 2
        eid = sprintf('Images:%s:invalidGrayLimitsSize',mfilename);
        msg = 'GL must be a two-element vector.';
        error(eid,'%s',msg);
      end
      gl = double(gl);
    
      case 'Symmetric'
        sym = varargin{idx};
        validateattributes(sym,{'logical'}, {'scalar'}, mfilename, 'Symmetric', idx);

      case 'Invariant'
        rotinv = varargin{idx};
        validateattributes(rotinv,{'logical'}, {'scalar'}, mfilename, 'Invariant', idx);
        
    end
  end
end
