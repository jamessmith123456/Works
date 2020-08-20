%%%%%%%VERSION 2
%%ANOTHER DESCRIBTION OF GABOR FILTER

%The Gabor filter is basically a Gaussian (with variances sx and sy along x and y-axes respectively)
%modulated by a complex sinusoid (with centre frequencies U and V along x and y-axes respectively) 
%described by the following equation
%%
%                            -1     x' ^     y'  ^             
%%% G(x,y,theta,f) =  exp ([----{(----) 2+(----) 2}])*cos(2*pi*f*x');
%                             2    sx'       sy'
%%% x' = x*cos(theta)+y*sin(theta);
%%% y' = y*cos(theta)-x*sin(theta);

%% Describtion :

%% I : Input image
%% Sx & Sy : Variances along x and y-axes respectively
%% f : The frequency of the sinusoidal function
%% theta : The orientation of Gabor filter

%% G : The output filter as described above
%% gabout : The output filtered image



%%  Author : Ahmad poursaberi  e-mail : a.poursaberi@ece.ut.ac.ir
%%          Faulty of Engineering, Electrical&Computer Department,Tehran
%%          University,Iran,June 2004

%Recent studies on Mathematical modeling of visual cortical cells [Kulikowski/Marcelja/Bishop:1982] 
%suggest a tuned band pass filter bank structure. These filters are found to have Gaussian transfer
%functions in the frequency domain. Thus, taking the Inverse Fourier Transform of this transfer 
%function we get a filter characteristics closely resembling to the Gabor filters. 
%The Gabor filter is basically a Gaussian (with variances sx and sy along x and y-axes respectively) 
%modulated by a complex sinusoid (with centre frequencies U and V along x and y-axes respectively).

%Gabor filters are used mostly in shape detectin and feature extractin in image processing.

%function [gabout, G] = gaborfilter1(I,Sx,Sy,f,theta); 
%from 'gaborfilter1' with different f(Frequency) and theta(Angle). 
%for example

%f:0,2,4,8,16,32 
%theta = 0,pi/3,pi/6,pi/2,3pi/4
%then for any input image like(eg. stereo.jpg)
%you have 6x5 = 30 filtered images.
%You can choose your desired angles or frequencies.
%You can put nominaly Sx & Sy = 2,4 or some one else.
%For instance I tested above example on ('cameraman.tif')(in MATLAB pictures)

%I = imread('cameraman.tif'); 
%[G,gabout] = gaborfilter1(I,2,4,16,pi/3); 
%figure,imshow(uint8(gabout));

function [gabout] = append_gaborfilter(I,lambda,theta)

bandwidth = 1;
gamma = 0.5;
sigma_x = lambda / pi * sqrt(log(2)/2)*(2^bandwidth+1)/(2^bandwidth-1);
sigma_y = sigma_x / gamma;
f = 1/lambda;

Sx = ceil(sigma_x);
Sy = ceil(sigma_y);

G_real = zeros(2*Sx, 2*Sy);
G_imag = zeros(2*Sx, 2*Sy);

for x = -Sx:Sx
    for y = -Sy:Sy
        xPrime = x * cos(theta) + y * sin(theta);
        yPrime = y * cos(theta) - x * sin(theta);
        
        G_real(Sx+x+1,Sy+y+1) = exp(-.5*((xPrime/sigma_x)^2+(yPrime/sigma_y)^2))*cos(2*pi*f*xPrime);
        G_imag(Sx+x+1,Sy+y+1) = exp(-.5*((xPrime/sigma_x)^2+(yPrime/sigma_y)^2))*sin(2*pi*f*xPrime);
    end
end

wNormalize = sqrt(sum(sum(G_real.*G_real+G_imag.*G_imag)));
G_real = G_real / wNormalize;
G_imag = G_imag / wNormalize;

if isa(I,'double')~=1 
    I = double(I);
end

Imgabout = conv2(I,double(G_imag),'same');
Regabout = conv2(I,double(G_real),'same');

gabout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);
end