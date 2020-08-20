function [video_file_path, fish_detection] = loadFishData(video_id, video_folder_path, mysql_folder_path, buse_cache )

    %addpath('../2012-03-16_MYSQL_connector/');

    fprintf('loading mysql data...');
    
    if buse_cache
        if ~exist(mysql_folder_path, 'dir');
            mkdir(mysql_folder_path);
        end
        
        [ mysql_exist, mysql_name, mysql_file_path ] = getFileNamefromID( video_id, mysql_folder_path );
        if ~mysql_exist
            fprintf('cannot find cached file. downloading...\n');
            [fish_detection] = queryFishDetectionMYSQL(video_id);
            save(fullfile(mysql_folder_path, fish_detection.videos.video_namewithoutext), 'fish_detection');
        else
            fprintf('find cached file. reading...');
            mysql_data = load(mysql_file_path);
            fish_detection = mysql_data.fish_detection;
        end
    else
        fprintf('downloading...\n');
        [fish_detection] = queryFishDetectionMYSQL(video_id);
    end
    
    fprintf('finished\n');

    [ video_exist, video_name, video_file_path ] = getFileNamefromID( video_id, video_folder_path );
    fprintf('loading video...');
    if ~video_exist
        if ~exist(video_folder_path, 'dir');
            mkdir(video_folder_path);
        end
        fprintf('cannot find video file. downloading...');
        video_name = fish_detection.videos.video_name;
        video_file_path = fullfile(video_folder_path, video_name);
        urlwrite(fish_detection.videos.url, video_file_path);
    end
    fprintf('finished\n');  
end

