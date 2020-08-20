function [ result ] = append_testNodeSplit( feature, featurei, classid, traj_id, cv_id, class_set, range_start, range_end, filesavename)

b_binary_split = 1;

result.range_start = range_start;
result.range_end = range_end;
result.class_set = class_set;

class_size = length(class_set);
if class_size <= 4 || range_end < range_start || range_start < 0
    return;
end

class_size_left = floor(class_size / 2);
class_subset_index = ismember(classid, class_set);
feature_subset = feature(class_subset_index, :);
classid_subset = classid(class_subset_index);
cv_id_subset = cv_id(class_subset_index);
traj_id_subset = traj_id(class_subset_index);

loop_range = range_start:range_end;
loop_size = length(loop_range);

result.class_size = class_size;
result.score_set = zeros(loop_size, 3);
result.binary_pattern = cell(loop_size, 1);
result.processed = 0;
result.time_start = zeros(loop_size, 6);
result.time_end = zeros(loop_size, 6);
result.time_tic = zeros(loop_size,1,'uint64');
result.time_toc = zeros(loop_size,1);

if exist(filesavename,'file')
    tmp_data = load(filesavename);
    if isfield(tmp_data, 'result') && tmp_data.result.range_start == range_start && tmp_data.result.range_end == range_end && isequal(tmp_data.result.class_set, class_set)
        result = tmp_data.result;
    end
end

for index_index = result.processed+1:loop_size
    index_num = loop_range(index_index);
    
    binary_pattern = de2bi(index_num, class_size) == 1;
    if b_binary_split && class_size_left ~= sum(binary_pattern)
        continue;
    end
    
    fprintf('%s processing %d/%d\n', append_timeString(), index_index, loop_size);
    result.time_start(index_index,:) = clock();
    result.time_tic(index_index) = tic;
    
    tmp_classid = ones(length(classid_subset), 1);
    tmp_class_indic = ismember(classid_subset, class_set(binary_pattern));
    tmp_classid(tmp_class_indic) = 0;
    
    tmp_result = classify_crossValidation(feature_subset, featurei, tmp_classid, traj_id_subset, cv_id_subset, 2, 0, 0, 0);
    result.score_set(index_index, 1) = tmp_result.cv_count_recall;
    result.score_set(index_index, 2) = tmp_result.cv_class_precision;
    result.score_set(index_index, 3) = tmp_result.cv_class_recall;
    
    result.binary_pattern{index_index} = binary_pattern;
    result.time_toc(index_index) = toc(result.time_tic(index_index));
    result.time_end(index_index,:) = clock();
    result.processed = index_index;
    save(filesavename, 'result');
    fprintf('...done\n');
end

end

