%% Loading the images from the database and converting the images into column matrix
clear all
clc
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
    m_test=mean(B);
    mean_mat_test = [mean_mat_test m_test];
    % Subtract the mean from each image [Centering the data]
    d_test=S-uint8(repmat(m_test,2576,1));
    d_test=double(d_test);

end
m_train=mean(C,2);
m_test=mean(T,2);
train_data=C-uint8(repmat(m_train,1,200));
test_data=T-uint8(repmat(m_test,1,200));

x=vec'*(double(train_data));
v_test=vec'*(double(test_data));

%To know what is the class of this test data use any classifier 
D=pdist2(v_test',x','euclidean');
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
ezroc3_(D,labels,2,'',1);
