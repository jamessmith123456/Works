function [ result ] = RecognizeFish( fish_task )

video_path = fish_task.video_path;
mysql_cache_path = 'mysql';
if fish_task.mysql_cache
    mysql_cache_path = fish_task.mysql_cache_path;
end

b_use_trajvote = 1;

video_num = length(fish_task.video_info);
result.by_video(video_num,1).predict = {};
result.by_frame = [];
for j = 1:video_num
    tmp_video_info = fish_task.video_info(j);
    tmp_video_id = tmp_video_info.video_id;
    
    [video_file_path, fish_detection] = loadFishData(tmp_video_id, video_path, mysql_cache_path, fish_task.mysql_cache);
    fish_number = length(tmp_video_info.fish_id);
    fprintf('procossing video %d/%d : %d fish.\n', j, video_num, fish_number);
    
    result.by_video(j).predict = cell(fish_number, 3);
    result.by_video(j).video_id = tmp_video_id;

    for i = 1:fish_number
        fprintf('reading data\t');
        
        tmp_fish_id = tmp_video_info.fish_id(i);
        
        [rgbimg, mskimg] = segmentFishDetection(video_file_path, fish_detection, tmp_fish_id);
        frame_id = fish_detection.frame_id(fish_detection.fish_id==tmp_fish_id);
        frame_num = length(frame_id);
 
        detection_number = length(rgbimg);
        
        fprintf('video: %d/%d ; fish: %d/%d; detection: %d.\n', j, video_num, i, fish_number, detection_number);

        tmp_predict = internal_recognizefish(rgbimg, mskimg);
        if b_use_trajvote
            [ tmp_predict ] = result_trajvote_single( tmp_predict);
        end
        
        result.by_video(j).predict(i,1) = {tmp_fish_id};
        result.by_video(j).predict(i,2) = {frame_id};
        result.by_video(j).predict(i,3) = {tmp_predict};
        
        record_start = size(result.by_frame, 1) + 1; record_end = record_start + frame_num - 1;
        result.by_frame(record_start:record_end,1)=tmp_video_id;
        result.by_frame(record_start:record_end,2)=tmp_fish_id;
        result.by_frame(record_start:record_end,3)=frame_id;
        result.by_frame(record_start:record_end,4)=tmp_predict;
        
        if fish_task.save_image
            internal_saveImage(tmp_video_id, frame_id, tmp_fish_id, rgbimg, mskimg, tmp_predict, fish_task.save_image_path);
        end
    end
    save('recognizefish', 'result');
end

end

function [ species_id ] = internal_recognizefish( rgbimg, mskimg )
    
    features = feature_generateFeatureSet(rgbimg, mskimg);
    species_id = classify_HierSVM_predict(features);
    fprintf('classification finished.\n');
end

function internal_saveImage(video_id, frame_id, fish_id, rgbimg, mskimg, predict_id, save_image_path)
    save_file_folder = [save_image_path '/' num2str(video_id)];
    if ~exist(save_file_folder, 'dir')
        mkdir(save_file_folder);
    end
    
    image_num = length(rgbimg);
    for i = 1:image_num
        save_file_path = [save_file_folder '/' num2str(fish_id) '_' num2str(frame_id(i), '%03d') '_' num2str(predict_id(i)) '.png'];
        image_fus = internal_fusion(rgbimg{i}, mskimg{i});
        imwrite(image_fus, save_file_path);
    end
end

function [ rgbimg ] = internal_fusion(rgbimg, mskimg)
    contour = bwperim(mskimg,4);
    rgbimg(contour)=255;
end
