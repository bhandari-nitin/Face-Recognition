clear all
clc

No_of_classes = 40;
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

%% PCA
data=C;
[row,col]=size(data);
% Compute the mean of the data matrix "The mean of each row"
m=(mean(data'))'; 

% Subtract the mean from each image [Centering the data]
d=data-uint8(repmat(m,1,200));
d=double(d);
% Compute the covariance matrix (co)
co=d*d';

% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);

% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
% eigvalue = eigvalue(index);
% eigvector = eigvector(:, index);

%% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complexity

vec=eigvector(:,index(1:200));

% Compute the feature matrix (the space that will use it to project the testing image on it)
x=vec'*d;

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
disp(size(T));
T=double(T);
m_test=(mean(T'))';
T=T-repmat(m_test,1,200);

%Project the testing data on the space of the training data
x_test=vec'*T;

%To know what is the class of this test data use any classifier 
D=pdist2(x_test',x','euclidean');

norm=max(D(:));
norm_mat=1/norm*(D);
euc=pdist2(x_test',x');
m=max(euc(:));
euc=1/m*euc;
norm_mat=norm_mat';
j=1;
for i=0:5:195
score_train_PCA(:,j)=mean(norm_mat(:,i+1:i+5),2); %to find the scores of PCA at each level
j=j+1;
end;
euc=euc';
 
j=1;
for i=0:5:195
score_test_PCA(:,j)=mean(euc(:,i+1:i+5),2);
j=j+1;
end;

%%LDA
%% Loading the images from the database and converting the images into
%% column matrix
No_of_classes = 40;
Images_per_class = 10;
Training_images = 5;

C = [];
Sw = zeros(2576);
cov_mat = [];
mean_mat = [];
Sb=0;
for i = 1:No_of_classes
    fn = cd(['C:\Users\Ritika\Desktop\Supervised Learning\Project_2\s' num2str(i)]);
    A = [];
    B = [];
    m = [];
    d = [];
    co = [];
   
    for j = 1:Training_images
        filename = [num2str(j) '.pgm'];
        img = imread(filename);
        A = imresize(img,0.5);
        A = (A(:));
        B = [B A];
      
    end
    C = [C B];
    %Finding the mean of each class
    m=mean(B,2);
    mean_mat = [mean_mat m];
    % Subtract the mean from each image [Centering the data]
    d=B-uint8(repmat(m,1,5));
    d=double(d);
    % Compute the covariance matrix (cov)
    co=d*d';
    Sw = Sw+co;
end

%Calculating the inverse of within class scatter
invSw=pinv(Sw);

%Calculating the overall mean of all the classes
m_overall=mean(mean(mean_mat));

%Calculating the between class variance
sb=mean_mat-repmat(m_overall,2576,40);
for i=1:size(sb,2)
    Sb=Sb+((sb(i))'*sb(i));
end

v=invSw*Sb;

%find eigen values and eigen vectors of v
[evec,eval]=eig(v);

% Sort eigen vectors according to eigen values (descending order) and select eigen vectors according to eigen values to generate fisher space
eigvalue = diag(eval);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = evec(:, index);

%% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complexity

vec=evec(:,index(1:40));

% Compute the feature matrix (the space that will use it to project the testing image on it)


%% If you have test data do the following

T = [];
Sw_test = zeros(2576);
cov_mat_test = [];
mean_mat_test = [];
Sb_test = 0;
for i = 1:No_of_classes
    fn = cd(['C:\Users\Ritika\Desktop\Supervised Learning\Project_1\s' num2str(i)]);
    R = [];
    S = [];
    m_test = [];
    d_test = [];
    co_test = [];
    for j = 6:10
        filename = [num2str(j) '.pgm'];
        img = imread(filename);
        R = imresize(img,0.5);
        R = (R(:));
        S = [S R];
    end
    T = [T S];
    %Finding the mean of each class
    m_test=mean(S,2);
    mean_mat_test = [mean_mat_test m_test];
    % Subtract the mean from each image [Centering the data]
    d_test=S-uint8(repmat(m_test,1,5));
    d_test=double(d_test);

end
m_train=mean(C,2);
m_test=mean(T,2);
train_data=C-uint8(repmat(m_train,1,200));
test_data=T-uint8(repmat(m_test,1,200));

v_train=vec'*(double(train_data));
v_test=vec'*(double(test_data));

%To know what is the class of this test data use any classifier 
D=pdist2(v_test',v_train','euclidean');

norm=max(D(:));
normmat=1/norm*(D);
euc=pdist2(v_test',v_train');
m=max(euc(:));
euc=1/m*euc;
normmat=normmat';
j=1;
for i=0:5:195
score_train_LDA(:,j)=mean(normmat(:,i+1:i+5),2); %%to find the scores of LDA at each level
j=j+1;
end;
euc=euc';
j=1;
for i=0:5:195
score_test_LDA(:,j)=mean(euc(:,i+1:i+5),2);
j=j+1;
end;

%%Merge Code(Multi classifier) 
%Scores sum of both PCA and LDA

sum_score=score_train_LDA+score_train_PCA;
test_sum_score=score_test_LDA+score_test_PCA;
 
%performance evaluation plotting
temp=[ones(1,5),zeros(1,195)]';
for i=1:40
            target(:,i)=temp(:,:);
           temp=circshift(temp,5);    %% creating Target value%%
 
end;
 
temp1=[zeros(1,5),ones(1,195)]';
for i=1:40
        target1(:,i)=temp1(:,:);
       temp1=circshift(temp1,5);    %% creating Target value%%
end;
 
for i=1:200
     for j=1:40
         tempmat=[score_train_LDA(i,j),score_train_PCA(i,j)];
         max_score(i,j)=max(tempmat);%Max score rule
     end;
 end;
 for i=1:200
     for j=1:40
         tempmat=[score_train_LDA(i,j),score_train_PCA(i,j)];
         min_score(i,j)=min(tempmat);%Min score rule
     end;
 end;
 
roc_max=ezroc3(max_score,target,2,'MAXIMUM RULE',1);
roc_min=ezroc3(min_score,target,2,'MINIMUM RULE',1);
roc_average=ezroc3(sum_score,target,2,'SUM RULE',1);
roc_LDA=ezroc3(score_train_LDA,target,2,'LDA',1);
roc_PCA=ezroc3(score_train_PCA,target,2,'PCA',1);
 
figure(22), plot(roc_max(2,:),roc_max(1,:),'LineWidth',2),axis([-0.002 1 0 1.002]); hold on
 plot(roc_min(2,:),roc_min(1,:),'yellow','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;
 
 plot(roc_average(2,:),roc_average(1,:),'red','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;
 
 plot(roc_PCA(2,:),roc_PCA(1,:),'blue','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;
 
 plot(roc_LDA(2,:),roc_LDA(1,:),'green','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;
 
 legend('MAX RULE','MIN RULE','SUM RULE','PCA','LDA');