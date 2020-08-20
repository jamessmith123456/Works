function [ frames ] = readFrames( video_file, frame_list)

num_frames = length(frame_list);
frames = cell(num_frames, 1);
if nargin < 2 || isempty(frame_list)
    fprintf('Empty frame input\n');
    return;
end

if ~exist(video_file, 'file')
    fprintf('Not such file\n');
    return;
end

readObj = mmreader(video_file);
for i = 1:length(frame_list)
    tmp_frame = frame_list(i);
    frames(i) = {read(readObj, [tmp_frame tmp_frame])};
end

end

