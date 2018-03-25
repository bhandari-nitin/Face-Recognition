clear all;
clc;

%% importing LDA, PCA and target data
load score_test_LDA.mat
load score_test_PCA.mat
load target.mat
load target1.mat

%% ROC without Multi-instance
roc_LDA=ezroc3(score_test_LDA,target,2,'LDA',1);
roc_PCA=ezroc3(score_test_PCA,target,2,'PCA',1);

%% ROC with Multi-instance
roc_M_LDA=ezroc3(score_test_LDA,target1,2,'LDA_M',1);
roc_M_PCA=ezroc3(score_test_PCA,target1,2,'PCA_M',1);


figure(),
hold on
plot(roc_M_LDA(2,:),roc_M_LDA(1,:),'red','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;

plot(roc_M_PCA(2,:),roc_M_PCA(1,:),'blue','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;

plot(roc_PCA(2,:),roc_PCA(1,:),'green','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;

plot(roc_LDA(2,:),roc_LDA(1,:),'yellow','LineWidth',2),axis([-0.002 1 0 1.002]); hold on;

legend('LDA with Multi-instance','PCA with Multi-instance','LDA without Multi-instance', 'PCA without Multi-instance');
