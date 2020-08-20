function area_ration = feature_MaskAreaRatio( mask_up,  mask_down)
    Area_up = regionprops(mask_up,'Area');
    Area_down = regionprops(mask_down,'Area');
    
    area_ration = 0;
    if length(Area_up) < 1 || length(Area_down) < 1
        return;
    end
	
    area_ration = Area_up(end).Area / max(1, Area_down(end).Area);

end