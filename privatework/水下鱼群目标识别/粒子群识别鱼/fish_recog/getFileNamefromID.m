function [ file_exist, file_name, file_path ] = getFileNamefromID( file_id, file_folder_path )

file_exist = 0;
file_name = [];
file_path = [];

files = dir(file_folder_path);
file_num = length(files);

find_file_name = {};
for i = 1:file_num
    if files(i).isdir
        continue;
    end
    
    find_file_name(end+1) = {files(i).name};
end

find_file_id = getVideoIDfromName(find_file_name);
for i = 1:length(find_file_id)
    tmp_index = find(file_id == find_file_id(i));
    if 0 < length(tmp_index)
        file_exist = 1;
        file_name = find_file_name{i};
        file_path = [file_folder_path '/' find_file_name{i}];

    end
end

end

