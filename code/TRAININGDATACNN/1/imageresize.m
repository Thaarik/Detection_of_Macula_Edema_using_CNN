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
