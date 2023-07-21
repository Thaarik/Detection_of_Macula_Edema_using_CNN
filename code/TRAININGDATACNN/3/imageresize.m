% % % % % % % multiple image resize 
% % dirData = dir('*.jpg'); 
% % fileNames = {dirData.name};     
% % for iFile = 1:numel(fileNames)  
% % newName = sprintf('%1d.jpg',iFile); 
% % movefile(fileNames{iFile},newName);        
% % end  
% % 
% % q=1;
% % p=1;
% % for i=1:10
% % oq=imread(strcat(num2str(q),'.jpg'));
% % odprz=imresize(oq,[336 448]);  %# specify your size here
% % imwrite(odprz,strcat(num2str(p),'.jpg'));
% % q=q+1;
% % p=p+1;
% % end 


contents=ls;
for i=1:length(contents)
try
   tempImage=imread(contents(i,:));
   tempImage=imresize(tempImage,[200,200]);
   extnposn=find(contents(i,:)=='.');
   extn=contents(i,extnposn+1:end);
   extn(extn==' ')=[];
   name=contents(i,:);
   name(name==' ')=[];
   imwrite(tempImage,name,extn)
catch
    %write err msg as
    %use msgbox
end
end
