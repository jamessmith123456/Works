function [ data_exist, video_file_path, mysql_file_path] = CheckData( video_id, video_folder_path, mysql_folder_path, bdownload )

    [ video_exist, video_name, video_file_path ] = getVideoNamefromID_file( video_id, video_folder_path );

    [ mysql_exist, mysql_name, mysql_file_path ] = getVideoNamefromID_file( video_id, mysql_folder_path );
    
    data_exist = video_exist & mysql_exist;
    
    if ~bdownload
        return;
    end
    
    
end