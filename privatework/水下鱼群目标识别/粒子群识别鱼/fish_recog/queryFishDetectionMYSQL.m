function [fish] = queryFishDetectionMYSQL(VideoNumber)
%function [fish fish videos video_file valid] = queryFishDetectionMYSQL(VideoNumber,StorageDir,component_id)

%
% ing Dr B.J.Boom                      bboom@inf.ed.ac.uk
% Edinburgh, United Kingdom                    March 2011
% University of Edinburgh                            IPAB
%--------------------------------------------------------
% Xuan (Phoenix) Huang                Xuan.Huang@ed.ac.uk
% Edinburgh, United Kingdom                    March 2012
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%

% Check to make sure that we successfully connected
user = 'root';
password = 'Pa$$w0rd';
dbName = 'f4k_db';
host = 'f4k-db.ing.unict.it';
mysql( 'open', host, user, password);
mysql( ['use ' dbName]);

video_number = length(VideoNumber);
fish(video_number).fish_id = [];
for video_index = 1:video_number
    command_execute = ['select detection_id,fish_id,frame_id,asText(bounding_box),asText(contour),detection_certainty,tracking_certainty from fish_detection where video_id = ' num2str(VideoNumber(video_index))];
    [fish(video_index).detection_id fish(video_index).fish_id fish(video_index).frame_id bounding_box contour fish(video_index).detection_certainty fish(video_index).tracking_certainty]  = mysql(command_execute);
    fish_number = size(fish.fish_id,1);
    if  0 == fish_number
        disp(['Error: no fish detections for video: ' num2str(VideoNumber)]);
        continue;
    end
    
    fish(video_index).video_id = ones(fish_number, 1)*VideoNumber;
    
    for rr = 1:fish_number
        tmp_str = char(bounding_box{rr,1});
        coords = polygon_string(tmp_str);
        fish(video_index).bounding_box_x1(rr,1) = coords(1,1);
        fish(video_index).bounding_box_y1(rr,1) = coords(2,1);
        fish(video_index).bounding_box_x2(rr,1) = coords(1,3);
        fish(video_index).bounding_box_y2(rr,1) = coords(2,3);
        fish(video_index).bounding_box_w(rr,1) = coords(1,3)-coords(1,1);
        fish(video_index).bounding_box_h(rr,1) = coords(2,3)-coords(2,1);
        
        tmp_str = char(contour{rr,1});
        fish(video_index).contour{rr,1} = polygon_string(tmp_str);
    end
    
    fish_id_array = unique(fish(video_index).fish_id);
    for fish_index = 1:length(fish_id_array)
        tmp_fish_id = fish_id_array(fish_index);
        command_execute_fish = ['select component_id from fish where fish_id = ' num2str(tmp_fish_id)];
        [fish(video_index).component_id(tmp_fish_id == fish(video_index).fish_id)]  = mysql(command_execute_fish);
    end
    
    command_execute_video = ['select video_id,camera_id,date_time,frame_height,frame_width,frame_rate from videos where video_id = ' num2str(VideoNumber)];
    [videos.videos_id videos.camera_id videos.date_time videos.frame_height videos.frame_width videos.frame_rate] = mysql(command_execute_video);
    [videos.camera_id videos.video_number videos.location ]  = mysql(['select camera_id,video_number,location from cameras where camera_id = ' num2str(videos.camera_id)]);

    [Y, M, D, H, MN, S] = datevec(videos.date_time);
    time_string = [num2str(Y,'%04d') num2str(M,'%02d') num2str(D,'%02d') num2str(H,'%02d') num2str(MN,'%02d')];

    if videos.frame_width > 400
        vid_resolution = 2;
    else
        vid_resolution = 1;
    end
    
    videos.url = ['http://gad240.nchc.org.tw/tai/video_query/query_uuid.php?hid_site=' videos.location{1} '&hid_video=' num2str(videos.video_number) '&hid_time=' time_string '&hid_resolution=' num2str(vid_resolution) '&hid_codec=1&hid_fps=' num2str(videos.frame_rate)];

    videos.video_name =['video_' time_string '_' num2str(VideoNumber) '.flv']; 
    videos.video_namewithoutext =['video_' time_string '_' num2str(VideoNumber)]; 
    fish(video_index).videos = videos;
end

mysql('close');

end