function result = append_sumResult(result_i)
    number = length(result_i);
    
    result.confus = result_i{1}.confus;
    result.precision = result_i{1}.precision;
    
    result.correct = result_i{1}.correct;
    result.returnnum = result_i{1}.returnnum;
    result.classnum = result_i{1}.classnum;
    
    result.classrecall = result_i{1}.classrecall;
    result.recallAver = result_i{1}.recallAver;
    
    result.classprecision = result_i{1}.classprecision;
    result.precisionAver = result_i{1}.precisionAver;
    
    for i = 2:number
        result.confus = result.confus + result_i{i}.confus;
        result.precision = result.precision + result_i{i}.precision;
        
        result.correct = result.correct + result_i{i}.correct;
        result.returnnum = result.returnnum + result_i{i}.returnnum;
        result.classnum = result.classnum + result_i{i}.classnum;
        
        result.classrecall = result.classrecall + result_i{i}.classrecall;
        result.recallAver = result.recallAver + result_i{i}.recallAver;
        
        result.classprecision = result.classprecision + result_i{i}.classprecision;
        result.precisionAver = result.precisionAver + result_i{i}.precisionAver;
    end
    
    result.confus = result.confus / number;
    result.precision = result.precision / number;
    
    result.correct = result.correct / number;
    result.returnnum = result.returnnum / number;
    result.classnum = result.classnum / number;
    
    result.classrecall = result.classrecall / number;
    result.recallAver = result.recallAver / number;
    
    result.classprecision = result.classprecision / number;
    result.precisionAver = result.precisionAver / number;
end