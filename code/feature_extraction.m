%%% mecular edima detection using svm algorithms
%% clear the window
clc
clear
close all


a = dir('TRAININGDATA\1\*.tif');
FEATURE= [];
for k = 1:length(a)

%% open the input image
  file = a(k).name;

% [filename,pathname] = uigetfile('dataset\1\*.jpg');
% im = imread([pathname,filename]);
   
x=imread(fullfile('TRAININGDATA\1\',file));

x=imresize(x,[256 256]);
%% Convert it to grayscale.
g = rgb2gray(x);
figure, imshow(g); title('GrayScale Original Image');

%

% Showing the Individual individual channels for the colour Retina image
red = x;
green = x;
blue = x;
Red =x(:,:,1) ;
% subplot(2,2,1); imshow(x); ('RGB Image'); title('Original Image');
Green=x(:,:,2) ;
% subplot(2,2,2); imshow(Red); ('Red Components'); title('Red Colour Channel');
% subplot(2,2,3); imshow(Green); ('Green Components'); title('Green Colour Channel');
Blue=x(:,:,3);
% subplot(2,2,4); imshow(Blue); ('Blue Components');title('Blue Colour Channel');
%% adaptive thresholding 

bw1= imbinarize(g,'adaptive','ForegroundPolarity','dark','Sensitivity',0.5);
% figure,imshow(bw1);title('adaptive thresholding');
bw1= im2uint8(bw1);
comp=imcomplement(bw1);
% figure,imshow(comp);
 %% Clean (Denoise) the GrayScale Image using Median Filter
   u = medfilt2(Green);
%Adjust the contrast (Make it darker)
%    figure, imshow(u); title('MedFiltered Grayscale Image');
   du = imadjust (u,[0.1 0.9], []);
   

%[rows,cols] = find(du==1);
%    figure; imshow(du); title('A little darken Filtered Grayscale Image')
   %%
% Get the complement of the Binary Image using "imcomplement"
   fu = imcomplement(du);
%    figure; imshow(fu); title('complement')
  %%
  % Adaptive Histogram Equalization
  hist = adapthisteq(fu); %AdaptHISTEQ focus more on some area of picture unlike HISTEQ
%   figure; imshow(hist); title ('adjust');
  %% morphological operations 
  
  % Structuring Element;
  s = strel('ball',8,9); %here the radius and height are both 9
  % Morphological Open
  
  fu_Open = imopen(hist,s);
%   figure; imshow(fuOpen); title ('opening')
  se2 = strel('line', 1, 1);
  fuuOpen = imerode(fu_Open, se2);
%   figure, imshow(fuuOpen); title('disk');
  % Remove Optic Disk
  opDisk = hist - fuuOpen; 
%   figure; imshow(opDisk); title ('removing optical disc');
  %2D Median Filter
  medFilt = medfilt2(opDisk);
%   figure; imshow(medFilt); title ('medFilt');
  % imopen function
  
  backGround = imopen(medFilt,s); 
%   figure; imshow(backGround); title ('background');
  % Remove Background
  I2 = medFilt - backGround;  
%   figure; imshow(I2); title ('removing background');
  % Image Adjustment
  I3 = imadjust(I2);                         
  % Gray Threshold
  level = graythresh(I3);
   % Binarization
  bw = imbinarize(I3,level);
  % Morphological Open 
  bw = bwareaopen(bw,35);
  bw= im2uint8(bw);
%   figure,imshow(bw); title('vessel image');
    I4=comp-bw;
%   I4=imfill(I4);
%   figure,imshow(I4);
%% Create a label matrix from the image.
CC = bwconncomp(I4);
L = labelmatrix(CC);
%% Convert the label matrix into RGB image, using default settings.
RGB = label2rgb(L);
% figure,imshow(RGB)
%% 
% SE = strel('disk',5);
imopen=bwareaopen(I4,5);
% figure,imshow(imopen);
ll=imclearborder(imopen);
figure,imshow(ll);
ll=bwareaopen(ll,20);
figure,imshow(ll);


BWoutline = bwperim(ll);
Segout = x; 
Segout(BWoutline) = 255; 
% figure, imshow(Segout), title('outlined original image');


label=bwlabel(ll);
figure,imshow(label);

label=im2bw(label);


[out]=GLCM_Features1(im2double(ll));
c=cell2mat(struct2cell(out));
ceil(length(c)/12)
c=c';

FEATURE=[FEATURE;c];
end
health=FEATURE;
save health.mat health

