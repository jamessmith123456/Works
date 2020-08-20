function varargout = delphi_ahp_gui(varargin)
% DELPHI_AHP_GUI MATLAB code for delphi_ahp_gui.fig
%      DELPHI_AHP_GUI, by itself, creates a new DELPHI_AHP_GUI or raises the existing
%      singleton*.
%
%      H = DELPHI_AHP_GUI returns the handle to a new DELPHI_AHP_GUI or the handle to
%      the existing singleton*.
%
%      DELPHI_AHP_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DELPHI_AHP_GUI.M with the given input arguments.
%
%      DELPHI_AHP_GUI('Property','Value',...) creates a new DELPHI_AHP_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before delphi_ahp_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to delphi_ahp_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help delphi_ahp_gui

% Last Modified by GUIDE v2.5 24-Apr-2020 08:37:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @delphi_ahp_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @delphi_ahp_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before delphi_ahp_gui is made visible.
function delphi_ahp_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to delphi_ahp_gui (see VARARGIN)

% Choose default command line output for delphi_ahp_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes delphi_ahp_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = delphi_ahp_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
DiaryDelphi(1);
fid = fopen('logo1.txt');
tline = fgetl(fid);
list_cell={}
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
    list_cell = [list_cell;tline];
end
fclose(fid);
set(handles.edit2,'String', list_cell);
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
DiaryDelphi(2);
fid = fopen('logo2.txt');
tline = fgetl(fid);
list_cell={}
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
    list_cell = [list_cell;tline];
end
fclose(fid);
set(handles.edit2,'String', list_cell);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
DiaryDelphi(3);
fid = fopen('logo3.txt');
tline = fgetl(fid);
list_cell={}
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
    list_cell = [list_cell;tline];
end
fclose(fid);
set(handles.edit2,'String', list_cell);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global leixing;
all_zhibiao = {'网络的基本模式','网络的规模','网络的攻击性/窃密性','网络隐蔽性','网络可控性'};
if leixing==1
    load('mhahp1.mat');
    [a,b] = mhahphanshu(A,B1,B2,B3,B4,B5,leixing);
    new_result = {};
    new_result{1}='校园区域网络AHP分析';
    new_result{2}=[all_zhibiao{b(1)},'---',num2str(a(b(1)))];
    new_result{3}=[all_zhibiao{b(2)},'---',num2str(a(b(2)))];
    new_result{4}=[all_zhibiao{b(3)},'---',num2str(a(b(3)))];
    new_result{5}=[all_zhibiao{b(4)},'---',num2str(a(b(4)))];
    new_result{6}=[all_zhibiao{b(5)},'---',num2str(a(b(5)))];
    set(handles.edit2,'String', new_result);
end
if leixing==2
    load('mhahp2.mat');
    [a,b] = mhahphanshu(A,B1,B2,B3,B4,B5,leixing);
    new_result = {};
    new_result{1}='手机用户网络AHP分析';
    new_result{2}=[all_zhibiao{b(1)},'---',num2str(a(b(1)))];
    new_result{3}=[all_zhibiao{b(2)},'---',num2str(a(b(2)))];
    new_result{4}=[all_zhibiao{b(3)},'---',num2str(a(b(3)))];
    new_result{5}=[all_zhibiao{b(4)},'---',num2str(a(b(4)))];
    new_result{6}=[all_zhibiao{b(5)},'---',num2str(a(b(5)))];
    set(handles.edit2,'String', new_result);
end
if leixing==3
    load('mhahp3.mat');
    [a,b] = mhahphanshu(A,B1,B2,B3,B4,B5,leixing);
    new_result = {};
    new_result{1}='运营商数据中心网络AHP分析';
    new_result{2}=[all_zhibiao{b(1)},'---',num2str(a(b(1)))];
    new_result{3}=[all_zhibiao{b(2)},'---',num2str(a(b(2)))];
    new_result{4}=[all_zhibiao{b(3)},'---',num2str(a(b(3)))];
    new_result{5}=[all_zhibiao{b(4)},'---',num2str(a(b(4)))];
    new_result{6}=[all_zhibiao{b(5)},'---',num2str(a(b(5)))];
    set(handles.edit2,'String', new_result);
end   
    
    


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
global leixing;
leixing = str2num(get(handles.edit1, 'String'));


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
DiaryDelphi(2);
fid = fopen('logo2.txt');
tline = fgetl(fid);
list_cell={}
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
    list_cell = [list_cell;tline];
end
fclose(fid);
set(handles.edit2,'String', list_cell);
