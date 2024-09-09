clc;
clear;
close all;
format short g;
%% To start the execution,load one of the datasets from lines 6 to 28.
load ecoli1.mat           
% load ecoli2.mat    
% load ecoli3-O.mat                     %ecoli3
% load ecoli4.mat
% load ecoli-0-1-4-7_vs_5-6.mat
%load ecoli-0-3-4_vs_5.mat
% load ecoli-0-1-4-7_vs_2-3-5-6.mat
% load glass0.mat
% load glass-O.mat                     %glass0123456
% load shuttle-2_vs_5.mat
% load vehicle2.mat                    %vehicle2–1
% load vowel0.mat 
% load wisconsin.mat
% load new-thyroid1-O.mat              %new-thyroid1
% load page-blocks-1-3_vs_4.mat
% load pima.mat

% load segment0.mat   %Lines 23 and 24 run together, otherwise you will get an error.     
% data=[data(:,1:2),data(:,4:end)]; 

% load kddcup-buffer_overflow_vs_back.mat   %Lines 26 and 27 run together, otherwise you will get an error.
% data=[data(:,1),data(:,3:6),data(:,10),data(:,13:14),data(:,16:17),data(:,23:33),data(:,36:42)];

%% Draw a graph on data
tic;
[r,c]=size(data); 
% labels=data(:,end);
% data1=data(:,1:end-1);
% figure, scatter(data1(:,1), data1(:,2),[], labels,'filled');
% title('dataset');
% xlabel('x');
% ylabel('y');
% colormap([0 0 1;1 0 0])  %RGB for example R=1 0 0

%% Data normalization
normalize_data=zeros(r,c-1);
for   i=1:c-1
normalize_data(:,i)=(data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end
data=[normalize_data,data(:,end)];
data=unique(data,'rows');
labels=data(:,end);
data1=data(:,1:end-1);

%% Call function under_classify
minority_data=data(data(:,c)==1,:);
majority_data=data(data(:,c)==2,:);

[after_knn_Precision,after_knn_Recall,after_knn_Fmeasure,after_knn_G_means,after_knn_accuracy,...
 after_svm_Precision,after_svm_Recall,after_svm_Fmeasure,after_svm_G_means,after_svm_accuracy,...
 after_cart_Precision,after_cart_Recall,after_cart_Fmeasure,after_cart_G_means,after_cart_accuracy,...
 Std_Dev_knn_recall,Std_Dev_knn_precision,Std_Dev_knn_F_measure,Std_Dev_knn_G_means,Std_Dev_knn_accuracy,...
 Std_Dev_svm_recall,Std_Dev_svm_precision,Std_Dev_svm_F_measure,Std_Dev_svm_G_means,Std_Dev_svm_accuracy,...
 Std_Dev_cart_recall,Std_Dev_cart_precision,Std_Dev_cart_F_measure,Std_Dev_cart_G_means,Std_Dev_cart_accuracy]=...
                                                                                 under_classify(data); 
end_time2=toc;
disp([ ' Time = '   num2str(end_time2) ' seconds' ])
                                                                                                                                              