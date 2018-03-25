%data extraction using imageset
Data = imageSet('faces_att','recursive');
train_data=cell(1,200);
test_data=cell(1,200);
a = 1;
 for j=1:40     
     for i=1:5     %first five images of all 40 subjects(classes) for training
         X= read(Data(j),i);
         X=reshape(X,prod(size(X)),1);
         X=double(X);
         train_data{a} = X;
         a = a + 1;
     end;
 end;
 a=1;
  for j=1:40       %40 subjects
     for i=6:10    %last five images (6 to 10) of all 40 subjects for testing
         X= read(Data(j),i);
         X=reshape(X,prod(size(X)),1);
         X=double(X);
         test_data{a} = X;
         a = a + 1;
     end;
 end;
 
%%converting the cellarray to ordinary array or matrix
train_data=cell2mat(train_data); 
test_data=cell2mat(test_data);

m=mean(train_data,2);
 
 
% Calculating Mean of Each Class%%
j=1;
for i=0:5:195
    temp(:,j)=mean(train_data(:,i+1:i+5),2);
    m_class(:,i+1:i+5)=repmat(temp(:,j),[1,5]);    %% Calculating Mean of Each Class%%
    j=j+1;
end;
 
% Calculate the within class scatter (SW)
     temp1=zeros(10304,10304);
     wsca=zeros(10304,10304);
 for i =0:5:195
     temp1=(train_data(:,i+1:i+5)-m_class(:,i+1:i+5))*((train_data(:,i+1:i+5)-m_class(:,i+1:i+5))');
     wsca=temp1+wsca;                %%calculating with in scatter matrix%% 
 end;
 
 v=pinv(wsca); % Calculate the within class scatter (SW)
 
  %%Calculating between scatter matrix%%
  temp2=zeros(10304,10304);
  bsca=zeros(10304,10304);
 for i=1:40
     temp2=(temp(:,i)-m)*((temp(:,i)-m)');
     bsca=temp2+bsca;            %%Calculating between scatter matrix%%
 
  end;
 
% Subtract the mean from each image [Centering the data]
d=train_data-repmat(m,1,200); %for the training set
 
test_data=test_data-repmat(mean(test_data,2),1,200);% performing the mean of the test matrix and subtracting the mean from each image(centering the data)
 
% find eigen values and eigen vectors of the (v)
[evec,eval]=eig(v*bsca);
 
% Sort the eigen vectors according to the eigen values
eigvalue = diag(eval);
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
vec=evec(:,index(1:40)); %Number of principal components used
 
%projection
 
tr_pro=vec'*d; %train projection
 
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