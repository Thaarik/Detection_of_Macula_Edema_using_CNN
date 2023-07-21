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

x=imresize(x,[200 200]);

figure,imshow(x);title('resized image')



%% train the input images 
matlabroot='F:\code\code\code\training';

Data=imageDatastore(matlabroot,'IncludeSubfolders',true,'LabelSource','foldernames');
%% ------------ CREATE CNN LAYERS -------------------%%
%image input layer inputs images to a network 
layers=[imageInputLayer([200 200 3]) 
%      The layer convolves the input by moving the filters along the input vertically and horizontally and computing the dot product of the weights and the input, and then adding a bias term.
convolution2dLayer(5,20)
% threshouling layers
    reluLayer
% A max pooling layer performs down-sampling by dividing the input into rectangular pooling regions, and computing the maximum of each region.    
   maxPooling2dLayer(2,'stride',2)
   
       convolution2dLayer(5,20)
       
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
    
    msgbox('No exudates')
    
    
elseif output==2
    
    msgbox('Soft Exudates')
    

elseif output==3
    
    msgbox('Hard exudates')
   
    
end




