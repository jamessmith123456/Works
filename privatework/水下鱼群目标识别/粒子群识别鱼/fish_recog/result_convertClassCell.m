function result_predict = result_convertClassCell(prediction, convertTable)
result_predict = cell(length(prediction), 1);
for i = 1:length(prediction)
    result_predict{i} = zeros(length(prediction{i}), 1);
    indic = prediction{i} > 0;
    result_predict{i}(indic) = convertTable(prediction{i}(indic));
end
end