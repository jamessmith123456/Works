function [rgb_img, msk_img] = segmentFishDetection(video_file_path, fish_detection, fish_id)
%
% ing Dr B.J.Boom                      bboom@inf.ed.ac.uk
% Edinburgh, United Kingdom                    March 2011 
% University of Edinburgh                            IPAB
%--------------------------------------------------------
%

mov_properties = mmreader(video_file_path);
nFrames = mov_properties.NumberOfFrames;
vidHeight = mov_properties.Height;
vidWidth = mov_properties.Width;

for kk = 1 : 50
    frame = read(mov_properties, kk);
    if kk == 1
        mean_frame = double(frame);
    else
        mean_frame = mean_frame * (kk/ (kk + 1)) + double(frame) * (1/ (kk + 1));
    end
end

tmp_fish_indic = find(fish_detection.fish_id==fish_id);
frame_num = size(tmp_fish_indic,1);
rgb_img = cell(frame_num,1);
msk_img = cell(frame_num,1);
for k=1:frame_num
    
    mask = tmp_fish_indic(k);
    x_window = fish_detection.bounding_box_x1(mask);
    y_window = fish_detection.bounding_box_y1(mask);
    w_window = fish_detection.bounding_box_w(mask);
    h_window = fish_detection.bounding_box_h(mask);
    blob_window = fish_detection.contour(mask);
    
    kk = fish_detection.frame_id(mask);
%    frame = read(mov_properties, kk);
    frame = read(mov_properties, kk+1);

    %figure(1); imshow(frame);
    %set(h,'CData',frame_buf(kk).cdata); 
    %hold on;

    padding = round(max(h_window,w_window)/2);
%     plot([x_window,x_window + w_window,x_window + w_window,x_window,x_window],...
%         [y_window,y_window,y_window + h_window,y_window + h_window,y_window] ,'r');
    
    im_rows = (y_window-padding+1):(y_window+h_window+padding);
    im_cols = (x_window-padding+1):(x_window+w_window+padding);
    fish_rows = 1:size(im_rows,2);
    fish_cols = 1:size(im_cols,2);
    
    mask_rows = im_rows >= 1 & im_rows <= size(frame,1);
    mask_cols = im_cols >= 1 & im_cols <= size(frame,2);
    
    
    fish_image = zeros(size(im_rows,2),size(im_cols,2),3,'uint8');
    fish_image_prev = zeros(size(im_rows,2),size(im_cols,2),3);
    fish_image(fish_rows(mask_rows),fish_cols(mask_cols),:) = frame(im_rows(mask_rows),im_cols(mask_cols),:);
    fish_image_prev(fish_rows(mask_rows),fish_cols(mask_cols),:) = mean_frame(im_rows(mask_rows),im_cols(mask_cols),:);
    
    fish_mask = roipoly(zeros(h_window,w_window), blob_window{1}(1,:), blob_window{1}(2,:));
    
    fish_mask = padarray(fish_mask,[padding padding]);
    valid_pixel_mask = double(mask_rows') * double(mask_cols);
    [fish_image,fish_mask] = improveFishContour(fish_image,fish_mask,fish_image_prev,valid_pixel_mask);

           
%     hold off;
%     drawnow;
    
    rgb_img(k) = {fish_image};
    msk_img(k) = {fish_mask};
    mean_frame = mean_frame * (kk/ (kk + 1)) + double(frame) * (1/ (kk + 1));
end

end