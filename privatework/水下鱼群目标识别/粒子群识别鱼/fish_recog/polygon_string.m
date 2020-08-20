function coords = polygon_string(str)

begin_idx = strfind(str,'((');
space_idx = strfind(str,' ');
comma_idx = strfind(str,',');
end_idx = strfind(str,'))');
comma_idx = [comma_idx end_idx];


idx = begin_idx+1;
not_finished = 1;

coords = zeros(2,size(space_idx,2));
for iter = 1:size(space_idx,2)
   coords(1,iter) = str2num(str(idx+1:space_idx(iter)-1)); 
   coords(2,iter) = str2num(str(space_idx(iter)+1:comma_idx(iter)-1));
   idx = comma_idx(iter);
end