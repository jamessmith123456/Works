load('./cifar-10-batches-mat/batches.meta.mat');
num = 1;
for k=1:5
    matname = ['./cifar-10-batches-mat/data_batch_',num2str(k),'.mat'];
    load(matname);
    for i=1:10000
        img = data(i,:);
        img = reshape(img',[32,32,3]);
        name = ['.cifar/',label_names{labels(i)+1},'/',num2str(num),'.png'];
        num = num+1;
        %imwrite(img,['F:/juanji/cifar/',label_names{labels(i)},'/',num2str(num),'.png']);
        imwrite(img,name);
    end
end
load('./cifar-10-batches-mat/test_batch.mat');
for i=1:10000
    img = data(i,:);
    img = reshape(img',[32,32,3]);
    name = ['./cifar/',label_names{labels(i)+1},'/',num2str(num),'.png'];
    num = num+1;
    %imwrite(img,['F:/juanji/cifar/',label_names{labels(i)},'/',num2str(num),'.png']);
    imwrite(img,name);
end



    
    