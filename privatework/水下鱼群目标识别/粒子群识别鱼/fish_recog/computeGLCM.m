%-----------------------------------------------------------------------------
function oneGLCM = computeGLCM(r,c,offset,si1,si2,nl)
% computes GLCM given one Offset

  r2 = r + offset(1);
  c2 = c + offset(2);
  
  % Remove pixel and its neighbor if they have subscripts outside the image
  % boundary.
  s = size(si1);
  bad = find(c2 < 1 | c2 > s(2) | r2 < 1 | r2 > s(1));
  Index = [r c r2 c2];
  Index(bad,:) = []; %#ok
  
  % Create vectors containing the values of r and c (v1 and v2 respectively) for
  % each pixel pair.
  v1 = si1(sub2ind(s,Index(:,1),Index(:,2)));
  v2 = si2(sub2ind(s,Index(:,3),Index(:,4)));
  
  % Make sure that v1 and v2 are column vectors.
  v1 = v1(:);
  v2 = v2(:);
  
  % Remove pixel and its neighbor if their value is NaN.
  bad = isnan(v1) | isnan(v2);
  if any(bad)
    wid = sprintf('Images:%s:scaledImageContainsNan',mfilename);
    msg = 'GLCM does not count pixel pairs if either of their values is NaN.';
    warning(wid,'%s', msg);
  end
  Ind = [v1 v2];
  Ind(bad,:) = [];
  
  if isempty(Ind)
    oneGLCM = zeros(nl);
  else
    % Tabulate the occurrences of pixel pairs having v1 and v2.
    oneGLCM = accumarray(Ind, 1, [nl nl]);
  end
  