%%% macular edema detection using cnn algorithms
%% clear the command window
clc
% clear the workspace
clear
% close the image viewer app
close all
warning off
%% open the input image
[filename,pathname] =uigetfile('DATASET\*.*');
x =imread([pathname,filename]);% READ THE IMAGE
figure; imshow(x); title('RGB Image');

%%to resize the input image%%

x=imresize(x,[256 256]);
%% Convert it to grayscale.

g = rgb2gray(x);

figure; imshow(g); title('GrayScale Original Image');

%%

% Showing the Individual individual channels for the colour Retina image
red = x;

green = x;

blue = x;

Red =x(:,:,1) ;
subplot(2,2,1); imshow(x); ('RGB Image'); title('Original Image');
Green=x(:,:,2) ;
subplot(2,2,2); imshow(Red); ('Red Components'); title('Red Colour Channel');
subplot(2,2,3); imshow(Green); ('Green Components'); title('Green Colour Channel');
Blue=x(:,:,3);
subplot(2,2,4); imshow(Blue); ('Blue Components');title('Blue Colour Channel');
%% adaptive thresholding 

bw1= imbinarize(g,'adaptive','ForegroundPolarity','dark','Sensitivity',0.5);

figure,imshow(bw1);title('adaptive thresholding');

bw1= im2uint8(bw1);

comp=imcomplement(bw1);

figure,imshow(comp);
 %% Clean (Denoise) the GrayScale Image using Median Filter
   u = medfilt2(Green);
   
%Adjust the contrast (Make it darker)
%    figure, imshow(u); title('MedFiltered Grayscale Image');
    du = imadjust (u,[0.1 0.9], []);
   

%[rows,cols] = find(du==1);
   figure; imshow(du); title('A little darken Filtered Grayscale Image')
   %%
% Get the complement of the Binary Image using "imcomplement"
   fu = imcomplement(du);
   figure; imshow(fu); title('complement')
  %%
  % Adaptive Histogram Equalization  CONTARST INCREASE
  hist = adapthisteq(fu); %AdaptHISTEQ focus more on some area of picture unlike HISTEQ
  
  figure; imshow(hist); title ('adjust');
  %% morphological operations 
  
  % Structuring Element;
  se = strel('ball',8,9); %here the radius and height are both 9
  % Morphological Open
  
  fuOpen = imopen(hist,se);
  
  figure; imshow(fuOpen); title ('opening')
  se2 = strel('line', 1, 1);
  
  fuuOpen = imerode(fuOpen, se2);
  figure, imshow(fuuOpen); title('disk');
  % Remove Optic Disk
  opDisk = hist - fuuOpen; 
  
  figure; imshow(opDisk); title ('removing optical disc');
  %2D Median Filter
  medFilt = medfilt2(opDisk);
  
  figure; imshow(medFilt); title ('medFilt');
  % imopen function
  backGround = imopen(medFilt,strel('disk',10)); 
  
  figure; imshow(backGround); title ('background');
  % Remove Background
  I2 = medFilt - backGround;  
  
  figure; imshow(I2); title ('removing background');
  % Image Adjustment
  I3 = imadjust(I2); 
  %% OTSU SEGMENTATION
  % Gray Threshold B/W level = 0.4941
  level = graythresh(I3);
  
   % Binarization
  bw = imbinarize(I3,level);
  figure,imshow(bw),title('image')
  % Morphological Open 
  bw = bwareaopen(bw,35);
  
  bw= im2uint8(bw);
  figure,imshow(bw); title('vessel image');
    I4=comp-bw;
    
%   I4=imfill(I4);       
  figure,imshow(I4);
%% Create a label matrix from the image.
CC = bwconncomp(I4);

L = labelmatrix(CC);
%% Convert the label matrix into RGB image, using default settings.
RGB = label2rgb(L);

figure,imshow(RGB);title('label the image')
%% 
SE = strel('disk',2,6);

imopen=imopen(I4,SE);
figure,imshow(imopen);
ll=imdilate(imopen,se);

figure,imshow(ll)
ll=imclearborder(imopen);

figure,imshow(ll);title('segmented')

BWoutline = bwperim(ll);

Segout = x; 

Segout(BWoutline) = 255; 

figure, imshow(Segout), title('outlined original image');
%%
%%%%------------------------------------------------------------------%%%%

[out]=GLCM_Features1(im2double(ll));

c=cell2mat(struct2cell(out));

ceil(length(c)/12)

c=c';
 % classification
load health;
% load moderate
load severe

feature=[health;severe];

train=ones(2,1);

train(1:1)=2;

train(2:end) = 1;
%[XL,YL,XS,YS,BETA,PCTVAR] = plsregress(feature,train',2);
%plot(1:2,cumsum(100*PCTVAR(1,:)),'-bo');
%xlabel("Number");
%ylabel("component");
%% support vector machine classification
Md1=fitcsvm(feature,train);

yfit=predict(Md1,c);
if yfit==1
    msgbox('healthy');
elseif yfit==2
    msgbox('disease go to checkup');
end

%%%%----------------------------------------------------------------------%%%%%%%



%% train the input images 
matlabroot='F:\code\code\code\TRAININGDATACNN';

Data=imageDatastore(matlabroot,'IncludeSubfolders',true,'LabelSource','foldernames');
%% ------------ CREATE CNN LAYERS -------------------%%
%image input layer inputs images to a network 
layers=[imageInputLayer([200 200 3]) 
%      The layer convolves the input by moving the filters along the input vertically and horizontally and computing the dot product of the weights and the input, and then adding a bias term.
convolution2dLayer(5,20)
% threshouling layers
batchNormalizationLayer
    reluLayer
% A max pooling layer performs down-sampling by dividing the input into rectangular pooling regions, and computing the maximum of each region.    
   maxPooling2dLayer(2,'stride',2)
   
       convolution2dLayer(5,20)
       batchNormalizationLayer
     reluLayer
     
 maxPooling2dLayer(2,'stride',2)
 
%  convolution2dLayer(5,20)
%     reluLayer
%    maxPooling2dLayer(2,'stride',2)
%A fully connected layer multiplies the input by a weight matrix and then adds a bias vector.
 fullyConnectedLayer(3)
%  A softmax layer applies a softmax function to the input.
 softmaxLayer
%  Create classification output layer
 classificationLayer()]
% Options for training deep learning neural network
options=trainingOptions('sgdm','MaxEpochs',15,'initialLearnRate',0.0001,'Plots','training-progress');
%Train neural network for deep learning
convnet=trainNetwork(Data,layers,options);
% CLASSIFICATION 
x=imresize(Segout,[200 200]);
% Classify data using a trained deep learning neural network
output=classify(convnet,x);

tf1=[];

for ii=1:3
    
    st=int2str(ii);
    
    tf=ismember(output,st);
%     Array elements that are members of set array
    tf1=[tf1 tf];
    
end
output=find(tf1==1);

if output==1
    
    msgbox('mild')
    y=imread('Mild.png');
    imshow(y);
    
elseif output==2
    
    msgbox('moderate')
    t=imread('Moderate.png');
    imshow(t);

elseif output==3
    
    msgbox('severe')
    z=imread('Severe.png');
    imshow(z);
    
end




clear