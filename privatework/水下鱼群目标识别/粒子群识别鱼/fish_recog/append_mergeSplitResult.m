function [ mergedResult, binaryPattern ] = append_mergeSplitResult( total )

mergedResult = [];
binaryPattern = {};
for id = 1:total
    filesavename = sprintf('nodeSplit_%04d_%d.mat', id, total);
    
    if exist(filesavename, 'file')
        load(filesavename, 'result');
        mergedResult(result.range_start:result.range_end,:)=result.score_set;
        binaryPattern(result.range_start:result.range_end,:) = result.binary_pattern;
    end
end


end

