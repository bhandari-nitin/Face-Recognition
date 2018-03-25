% %data extraction using imageset
% Data = imageSet('C:\Users\Ritika\Desktop\Supervised Learning\Project_1\','recursive');
% train_data=cell(1,200);
% test_data=cell(1,200);
% a = 1;
%  for j=1:40     
%      for i=1:5     %first five images of all 40 subjects for training
%          X= read(Data(j),i);
%          X=reshape(X,prod(size(X)),1);
%          X=double(X);
%          train_data{a} = X;
%          a = a + 1;
%      end;
%  end;
%  a=1;
%   for j=1:40       %40 subjects
%      for i=6:10    %last five images (6 to 10) of all 40 subjects for testing
%          X= read(Data(j),i);
%          X=reshape(X,prod(size(X)),1);
%          X=double(X);
%          test_data{a} = X;
%          a = a + 1;
%      end;
%  end;
%% Loading the images from the database and converting training images into column matrix
clear all
clc
No_of_classes = 40;
Images_per_class = 10;
Training_images = 5;

A = [];
C = [];

for i = 1:No_of_classes
    fn = cd(['C:\Users\Ritika\Desktop\Supervised Learning\Project_1\s' num2str(i)]);
    B = [];
    for j = 1:Training_images
        filename = [num2str(j) '.pgm'];
        img = imread(filename);
        A = imresize(img,0.7);
        A = (A(:));
        B = [B A];
    end
    C = [C B];
end

%% If you have test data do the following
R = [];
T = [];
for i = 1:No_of_classes
    fn = cd(['C:\Users\Ritika\Desktop\Supervised Learning\Project_1\s' num2str(i)]);
    S = [];
    for j = 6:10
        filename = [num2str(j) '.pgm'];
        img = imread(filename);
        R = imresize(img,0.7);
        R = (R(:));
        S = [S R];
    end
    T = [T S];
end

%%converting the cellarray to ordinary array or matrix
train_data=cell2mat(train_data); 
test_data=cell2mat(test_data);
 
% Compute the mean of the data matrix
m=mean(train_data,2); %for the training set
 
% Subtract the mean from each image [Centering the data]
d=train_data-repmat(m,1,200); %for the training set
 
% Compute the covariance matrix (co)
co=d*d';
 
% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);
 
 
% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
 
 
% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end
 
 
% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complixity
vec=eigvector(:,index(1:200));
 
%%projection
 
tr_pro=vec'*d; %train projection
 
test_data=test_data-repmat(mean(test_data,2),1,200);% performing the mean of the test matrix and subtracting the mean from each image(centering the data)
ts_pro=vec'*test_data; %test projection
 
 
%Use Euclidean distance as distance metrics.
 
D=pdist2(tr_pro',ts_pro','Euclidean');
 
%labels 
labels=zeros(200,200);
for i=1:200
    for j=1:200
        if(fix((i-1)/5)==fix((j-1)/5))
            labels(i,j)=0;
        else
            labels(i,j)=1;
        end
    end
end
 
%performance evaluation plotting
ezroc3(D,labels,2,'',1);
Chat Conversation End
