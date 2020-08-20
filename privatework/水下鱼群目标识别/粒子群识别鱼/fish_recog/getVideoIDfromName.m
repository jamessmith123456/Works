function [ video_id ] = getVideoIDfromName( video_name )

video_num = length(video_name);
video_id = zeros(video_num,1);
for i = 1:video_num
tmp_name = regexp(video_name{i},'\.','split');
tmp_name = tmp_name{1};
tmp_name = regexp(tmp_name,'_','split');
tmp_name = tmp_name{3};

video_id(i,1)=str2double(tmp_name);
end


end

