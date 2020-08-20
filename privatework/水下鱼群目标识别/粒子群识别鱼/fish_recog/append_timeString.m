function [ timeString ] = append_timeString()

ctime = clock();
timeString = sprintf('%d:%d:%d:%d:%d:%f', ctime(1), ctime(2), ctime(3), ctime(4), ctime(5), ctime(6));



end

