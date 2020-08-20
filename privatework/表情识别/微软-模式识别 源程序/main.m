%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%main����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;
close all;
warning off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%��������MAT�ļ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%//////////////////////////////////////////////////////////////////////////
%����������Ϣ����������ԭʼ�ľ������ֶ�¼�������γɣ��������
load('mat_SamplesInfo.mat'); %�������Samples����213*5
fprintf ('<����> ԭʼ������Ϣ����...\n\n');

%//////////////////////////////////////////////////////////////////////////
%�������ݾ��󣬸þ�������ÿ��ѵ��������3��ROI��ȫ������
if ~exist('mat_SamplesData.mat','file') %samples_Data��213*12��cell
    fprintf ('<����> �������ݾ���...\n\n');
    samples_Data=create_samplesData('JAFFE_ROI/','JAFFE_CUT/',Samples);%�����þ���
else
    load('mat_SamplesData.mat');
    fprintf ('<����> �������ݾ���...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%����Gabor�˲�����Ӧ����
if ~exist('mat_Gabor.mat','file') %G����5*8��cell��ÿ��cell��32*32����
    fprintf ('<����> Gabor�˲�����Ӧ����...\n\n');
    G=create_Gabor();%�����þ���
else
    load('mat_Gabor.mat');
    fprintf ('<����> Gabor�˲�����Ӧ����...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%����Gabor����ѵ�����ݼ�
% if ~exist('mat_trainSet.mat','file')||~exist('mat_trainSetP.mat','file')||~exist('mat_trainSetT.mat','file')
if ~exist('mat_trainSetP.mat','file')||~exist('mat_trainSetT.mat','file')    
    fprintf ('<����> Gabor����ѵ�����ݼ�...\n\n');
    [trainSet,P,T]=create_trainSet(G,samples_Data);
else
    askStr=input('�Ƿ����´���ѵ�����ݼ���(Y/N) :','s');
    if askStr=='Y'||askStr=='y'
        [trainSet,P,T]=create_trainSet(G,samples_Data);
    else
        load('mat_trainSetP.mat'); %P 354*213��double����    354������ά�ȣ�7�Ǳ��������
        load('mat_trainSetT.mat'); %T 7*213��double����
    end
    trainP=P;
    trainT=T;
    testP=P; %����ɶ��˼��
    testT=T;
    %�����ǹ�һ��������maxP minP�ֱ���P�����ֵ����Сֵ   maxT minT�ֱ���T�����ֵ����Сֵ 
    %PP��354*213����һ��֮���P  TT�ǹ�һ��֮���T,7*213 
    [PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);%������ֵ����Сֵ  
    fprintf ('<����> Gabor����ѵ�����ݼ�...\n\n');
end

%//////////////////////////////////////////////////////////////////////////
%���������������
if exist('mat_net.mat','file')
    fprintf ('<����> ���������������\n\n');
    load('mat_net.mat'); %Net���Ѿ��е������������
else
    fprintf ('<��ʾ> û���ҵ��������������������ѵ��\n\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���˵�ѭ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (1==1)
    choice=menu('        ����ʶ��           ',...
                '>>JAFFE�Բ���<<',...
                '>>ѵ��BP�����������<<',...
                '>>���Ա���ͼƬ<<',...
                '>>�ֶ������������<<',...
                '>>�ֶ�ROI����(�۾� ���� ���)����<<',...
                '>>KNN�������<<',...
                '>>�˳�<<');                

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %JAFFE�Բ���
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if (choice ==1)
        %//////////////////////////////////////////////////////////////////
        %������֤�����ַ�ʽ����ѵ�����ϺͲ��Լ���
        fprintf('���ַ�ʽ����ѵ�����Ͳ��Լ�:\n');
        fprintf('1�����ո�����࣬9������������Ϊѵ������1������������Ϊ���Լ�\n');
        fprintf('2�����ձ�����࣬���и���2/3�ĸ��������Ϊѵ������1/3�ĸ��������Ϊ���Լ�\n');
        askStr=input('��ѡ�񹹽���ʽ:','s');
        if askStr=='1'
            noStr=input('��������Ϊ���Եĸ������(1,2,...,10),��������������Ϊ���� :','s');
            %1��Ӧ1:23  2��Ӧ24:45  3��Ӧ46:67 4��Ӧ68:87 5��Ӧ88:108
            %6��Ӧ1:129:,7��Ӧ130:149,8��Ӧ150:170,9��Ӧ171:191,10��Ӧ192:213
            switch str2num(noStr)  
                case 1
                    startNo=1;
                    endNo=23;
                case 2
                    startNo=24;
                    endNo=45;
                case 3
                    startNo=46;
                    endNo=67;
                case 4
                    startNo=68;
                    endNo=87;
                case 5
                    startNo=88;
                    endNo=108;
                case 6
                    startNo=1;
                    endNo=129;
                case 7
                    startNo=130;
                    endNo=149;
                case 8
                    startNo=150;
                    endNo=170;
                case 9
                    startNo=171;
                    endNo=191;
                case 10
                    startNo=192;
                    endNo=213;
                otherwise
                    msgbox('������1-10֮���������Ĭ���Ե�10������������Ϊ�������ݣ�');
                    startNo=192;
                    endNo=213;
            end
            %��ȡ��������
            testP=P(:,startNo:endNo);
            testT=T(:,startNo:endNo);%��ǩ����
            %��ȡѵ������
            trainP=P;
            trainP(:,startNo:endNo)=[];%�ðɣ�ԭ���������ӵģ��������Բ���ȥ����ʣ�����Ϊѵ����
            trainT=T;
            trainT(:,startNo:endNo)=[];
            
        elseif askStr=='2'
            msgbox('��û������'); %666666���аɣ�����ʵ���Ͼ���1�����ݣ���΢�޸�һ��
%             for i=1:213
%                 if samples_Data{i,}
%             end
        else
            msgbox('ѡ�����������ѡ��');
            continue;
        end

       
        %//////////////////////////////////////////////////////////////////
        %��ʼ�Բ���
        fprintf('<��ʼ> JAFFE���ݼ��Բ���...\n\n');
        testP=tramnmx(testP,minP,maxP);%��һ�����ݼ�
        result=sim(Net,testP);%����
        result=postmnmx(result,minT,maxT);%��һ�����
        save mat_result result;%������
        fprintf('<����> JAFFE���ݼ��Բ���\n\n');
        
        %ͳ���Բ��Ե���ȷ��
        [m n]=size(testP);% mΪ354 nΪ22
        count=0;%ͳ����ȷ����������
        num=0;%ͳ�ƴ������������
        for i=1:n
            [a b]=sort(result(:,i),'descend');
            if T(b(1,1),i)==1
                count=count+1;
            else
                num=num+1;
                i;
            end
        end
        str='<����> �Բ��ԣ�������ȷ�� �� ';
        str=strcat(str,num2str(100*count/n));
        str=strcat(str,'%');
        disp(str);
    end    
    
    
    
                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %ѵ��������
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 2)
        %//////////////////////////////////////////////////////////////
        %�û�ѡ��
        isTrain=0;
        if exist('mat_net.mat','file')
            askStr=input('<��ʾ> ����BP��������������Ƿ����ѵ���µķ�������(Y/N) :','s');
            if askStr=='Y'||askStr=='y'
                isTrain=1;
            else
                isTrain=0;
            end   
        else
            isTrain=1;
        end
        
        %//////////////////////////////////////////////////////////////
        %�����Ƿ���ڷ��������û�ѡ������Ƿ�ѵ���µķ�����
        if isTrain==1
            %ѵ���µ������������
            fprintf ('<��ʼ> ѵ�������������...\n\n');
            fprintf('��ȫ��10������������Ϊѵ�����ݼ�������ȫ������Ϊ���Լ�\n');
            %ѵ��������
            [Net,minP,maxP,minT,maxT]=net_Train(trainP,trainT);
            fprintf ('<����> ѵ�������������\n\n');
        else
            fprintf ('<����> ���������������\n\n');
            load('mat_net.mat');
            [PP,minP,maxP,TT,minT,maxT]=premnmx(P,T);%������ֵ����Сֵ
        end
        trainP=P;
        trainT=T;
        testP=P;
        testT=T;
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %����ͼƬ
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 3)
        close all;
        %�жϷ������Ƿ��Ѿ�����
        if exist('Net','var')==0||exist('minP','var')==0||exist('maxP','var')==0
            msgbox('û���ҵ�����������������ȼ��ػ�ѵ����');
            continue;
        end
        
        %��ͼƬ�Ի���
        [file_name file_path] = uigetfile ('*.jpg');
        if file_path ~= 0
            %����ͼƬ
            str='<��ʾ> ��ȡͼƬ <';
            str=strcat(str,file_name);
            str=strcat(str,'> .....\n\n');
            fprintf(str);
            I=imread([file_path,file_name]);
            figure(1),subplot(121),imshow(I),title('����ͼƬԭͼ��');
            
            %��תͼƬ
            isGo=1;
            while(isGo==1)
                fprintf('\n');
                askStr=input('Ҫ��תͼ����(Y/N) :','s');
                if askStr=='Y'||askStr=='y'
                    fprintf('\n');
                    angleStr=input('��������ת�Ƕ�(0-180��,������Ӧ��ʱ��) :','s');
                    %��ת
                    I=imrotate(I,str2num(angleStr));
                    subplot(122),imshow(I),title('��ת��ͼƬ��');
                    isGo=1;
                else
                    isGo=0;
                end
            end
            
            %�и�ROI���и�õ�ROIͼƬ���浽TempĿ¼��
            fprintf('\n');
            askStr=input('Ҫ����ROI��ȡ��?(Y/N) :','s');
            if askStr=='Y'||askStr=='y'
                isComplete=0;
                while(isComplete==0)%ѭ����ȡ���̣�֪����ȡ��ϣ�Ϊ���Ǳ�����ȡ���̳�������������
                    fprintf('<��ʾ> ���ڽ���ROI�ָ�.....\n\n');
                    isComplete=getROI('mat',file_path,'Temp/',[],I,file_name);
                end
            end

            %��������������,�γ�һ��636*1�����ݾ���testSet��Ҳ��testP
            fprintf('\n<����> ���Բ�����������...\n\n');
            testSetP=create_testSet(G,'Temp/',file_name);
            
            %��������
            fprintf('\n<��ʼ> ������������...\n\n');
            testSetP=tramnmx(testSetP,minP,maxP);%�������ݹ�һ����ѵ�����ݷ�Χ
            testResult=sim(Net,testSetP);%�������з��������з���
            testResult=postmnmx(testResult,minT,maxT);%�����һ������ǩ��Χ
%             save mat_testResult testResult;%������

            %������
            label=cell(1,7);
            label{1,1}='����';            label{1,2}='����';            label{1,3}='�־�';
            label{1,4}='����';            label{1,5}='����';            label{1,6}='����';            label{1,7}='����';
            [a b]=sort(testResult(:,1),'descend');
            a=a*100;%��һ����100��
            
            str='<���> ������Ϊ�� \n';
            fprintf(str);
            %��һ���
            str='<1: ';
            str=strcat(str,num2str(a(1,1)));
            str=strcat(str,'->');
            fprintf(str);
            str=label{1,b(1,1)};
            str=strcat(str,'>\n\n');
            fprintf(str);
            %�ڶ����
            str='<2: ';
            str=strcat(str,num2str(a(2,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(2,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
            %�������
            str='<3: ';
            str=strcat(str,num2str(a(3,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(3,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
            %���Ľ��
            str='<4: ';
            str=strcat(str,num2str(a(4,1)));
            str=strcat(str,'->');
            str=strcat(str,label{1,b(4,1)});
            str=strcat(str,'>\n\n');
            fprintf(str);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %��������
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    if (choice ==4)
        msgbox('�Ѿ����кã��ݲ�����');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %����ROI����
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if (choice == 5)
        msgbox('�Ѿ����кã��ݲ�����');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %KNN����
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if (choice == 6)
        msgbox('�Ѿ����кã��ݲ�����');
    end    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�˳�
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (choice == 7)
        clear all;
        clc;
        close all;
        return;
    end    
    
end