function [Ave_precision_knn,Ave_recall_knn,Ave_F_measure_knn,Ave_G_means_knn,Ave_accuracy_knn,...
          Ave_precision_svm,Ave_recall_svm,Ave_F_measure_svm,Ave_G_means_svm,Ave_accuracy_svm,...
          Ave_precision_cart,Ave_recall_cart,Ave_F_measure_cart,Ave_G_means_cart,Ave_accuracy_cart,...
          Std_Dev_knn_recall,Std_Dev_knn_precision,Std_Dev_knn_F_measure,Std_Dev_knn_G_means,Std_Dev_knn_accuracy,...
          Std_Dev_svm_recall,Std_Dev_svm_precision,Std_Dev_svm_F_measure,Std_Dev_svm_G_means,Std_Dev_svm_accuracy,...
          Std_Dev_cart_recall,Std_Dev_cart_precision,Std_Dev_cart_F_measure,Std_Dev_cart_G_means,Std_Dev_cart_accuracy]=...
          under_classify(data)
%% Initialization of variables
[~,c]=size(data);
minority_data=data(data(:,c)==1,:);
majority_data=data(data(:,c)==2,:);
data=[minority_data;majority_data];

lable=data(:,end);
data1=data(:,1:end-1);

[no_class,~]=size(unique(lable));
kf=10;      %number of folds ,kfold=10
k=3;        %Number of neighbors for KNN classification 
kernel_svm='RBF';   %for SVM  
MaxNumSplits=7;     %for CART
iter=5;             %number of Iteration  

%% Define the variable to calculate the standard deviation *********
all_recall=zeros(kf,iter,3);
all_precision=zeros(kf,iter,3);
all_F_measure=zeros(kf,iter,3);
all_G_means=zeros(kf,iter,3);
all_accuracy=zeros(kf,iter,3);

for j=1:iter  
%% Define performance criteria before under-sampling*****************
recall_before=zeros(kf,3);
precision_before=zeros(kf,3);
F_measure_before=zeros(kf,3);
G_means_before=zeros(kf,3);
accuracy_before=zeros(kf,3);

%% Define performance criteria after under-sampling*****************
recall_after=zeros(kf,3);
precision_after=zeros(kf,3);
F_measure_after=zeros(kf,3);
G_means_after=zeros(kf,3);
accuracy_after=zeros(kf,3);

%% Definition of confusion matrix before and after under-sampling*******
xs_knn_before=zeros(no_class);
xs_svm_before=zeros(no_class);
xs_cart_before=zeros(no_class);

xs_knn_after=zeros(no_class);
xs_svm_after=zeros(no_class);
xs_cart_after=zeros(no_class);

%% Divide the data into k parts or rewriting Kfold approach*****************
[minority_index]=kfold_func2(minority_data,kf);
[majority_index]=kfold_func2(majority_data,kf);
indices=[minority_index;majority_index];

for i = 1:kf
%  tic;
%% Define variables for program tracing
   iter_num=j    
   foldi=i  
   test = (indices == i); train = ~test;
   
   d=data(train(),:);
   min_data_train=d(d(:,c)==1,:);
   maj_data_train=d(d(:,c)==2,:);
   
   t=data(test(),:);
   min_data_test=t(t(:,c)==1,:);
   maj_data_test=t(t(:,c)==2,:);
        
   %% create KNN model before under-sampling***************************
   Model_knn_before= fitcknn(data1(train(),:),lable(train(),:), 'NumNeighbors',k);
   predicted_label_knn_before = predict(Model_knn_before,data1(test(),:));
   xs_knn_before=confusionmat(lable(test(),:),predicted_label_knn_before);
   
   %% create SVM model before under-sampling**************************
   model_svm_before = fitcsvm(data1(train(),:),lable(train(),:),'Standardize',true,'KernelFunction',kernel_svm,'KernelScale','auto');
   predicted_label_svm_before = predict(model_svm_before,data1(test(),:));
   xs_svm_before=confusionmat(lable(test(),:),predicted_label_svm_before);
   
   %% create CART model before under-sampling**************************
   Model_cart_before=fitctree(data1(train(),:),lable(train(),:),'MaxNumSplits',MaxNumSplits);
   predicted_label_cart_before = predict(Model_cart_before,data1(test(),:));
   xs_cart_before=confusionmat(lable(test(),:),predicted_label_cart_before);
   
   %% KNN_Measures_before under-sampling************************
   [recall_before(i,1),precision_before(i,1),F_measure_before(i,1),G_means_before(i,1),accuracy_before(i,1)]=...
                                                               measures_of_classify(xs_knn_before);
                                                           
   %% SVM_Measures_before under-sampling************************
   [recall_before(i,2),precision_before(i,2),F_measure_before(i,2),G_means_before(i,2),accuracy_before(i,2)]=...
                                                               measures_of_classify(xs_svm_before);
                                                           
   %% CART_Measures_before under-sampling************************
   [recall_before(i,3),precision_before(i,3),F_measure_before(i,3),G_means_before(i,3),accuracy_before(i,3)]=...
                                                               measures_of_classify(xs_cart_before);
    
   %% Call one of the multi-manifold or single-manifold approaches. 
   % It means one of functions  multi_manifold and single_manifold  
   
   % Calling for a multi-manifold approach
    [sorted_weight,sorted_data,sort_lable]=multi_manifold(data(train(),:));
     
   % Calling for a single-manifold approach
   %[sorted_weight,sorted_data,sort_lable]=single_manifold(data(train(),:));
    
   %% undersampling stage*************************
   [under_weight,under_data]=data_deletion_func(sorted_weight,sorted_data,sort_lable); 
    
   %% create KNN model after under-sampling*******************************
   Model_knn_after= fitcknn(under_data(:,1:end-1),under_data(:,end), 'NumNeighbors',k);
   predicted_label_knn_after = predict(Model_knn_after,data1(test(),:));
   xs_knn_after=confusionmat(lable(test(),:),predicted_label_knn_after);
   
   %% create SVM model after under-sampling*************************************
   model_svm_after = fitcsvm(under_data(:,1:end-1),under_data(:,end),  'Standardize',true,'KernelFunction',kernel_svm,'KernelScale','auto');
   predicted_label_svm_after = predict(model_svm_after,data1(test(),:));
   xs_svm_after=confusionmat(lable(test(),:),predicted_label_svm_after);
   
   %% create CART model after under-sampling****************************************
   Model_cart_after = fitctree(under_data(:,1:end-1),under_data(:,end),'MaxNumSplits',MaxNumSplits);
   predicted_label_cart_after = predict(Model_cart_after,data1(test(),:));
   xs_cart_after=confusionmat(lable(test(),:),predicted_label_cart_after);
   
   %% KNN_Measures_after under-sampling************************
   [recall_after(i,1),precision_after(i,1),F_measure_after(i,1),G_means_after(i,1),accuracy_after(i,1)]=...
                                                               measures_of_classify(xs_knn_after);
                                                           
   %% SVM_Measures_after under-sampling***********************
   [recall_after(i,2),precision_after(i,2),F_measure_after(i,2),G_means_after(i,2),accuracy_after(i,2)]=...
                                                               measures_of_classify(xs_svm_after);
                                                           
   %% CART_Measures_after under-sampling***********************
   [recall_after(i,3),precision_after(i,3),F_measure_after(i,3),G_means_after(i,3),accuracy_after(i,3)]=...
                                                               measures_of_classify(xs_cart_after);
  
   %% while loop for KNN model - gradual under-sampling*******************
   under_data_knn=under_data;
   under_weight_knn=under_weight;
   
   while  F_measure_after(i,1)>=F_measure_before(i,1)    

    if F_measure_after(i,1)==1
       break;
    end   
    if  F_measure_after(i,1)>F_measure_before(i,1)     
        F_measure_before(i,1)=F_measure_after(i,1);  
    end
    sorted_weight=under_weight_knn(:,1);
    sorted_data=under_data_knn;
    sort_lable=under_data_knn(:,end);
    
   %under-sampling *************************
  [under_weight_knn,under_data_knn,five_percent]=data_deletion_func(sorted_weight,sorted_data,sort_lable);
  if five_percent==0
      break;
  end    
  
   %create KNN model after under-sampling*******************************************
   Model_knn_after= fitcknn(under_data_knn(:,1:end-1),under_data_knn(:,end), 'NumNeighbors',k);
   predicted_label_knn_after = predict(Model_knn_after,data1(test(),:));
   xs_knn_after=confusionmat(lable(test(),:),predicted_label_knn_after);
   [recall_after(i,1),precision_after(i,1),F_measure_after(i,1),G_means_after(i,1),accuracy_after(i,1)]=...
                                                               measures_of_classify(xs_knn_after);
   end
   if  F_measure_after(i,1)<F_measure_before(i,1)
       F_measure_after(i,1)=F_measure_before(i,1);
   end  
   %% Draw a graph after under-sampling with KNN model
   
%     b_data=under_data_knn(:,1:end-1);
%     b_lable=under_data_knn(:,end);
%     figure, scatter(b_data(:,1), b_data(:,2),[], b_lable,'filled');
%     title('under-sampling data');
%     xlabel('x1');
%     ylabel('x2');
%     colormap([0 0 1;1 0 0])  %RGB for example R=1 0 0

   
  %% while loop for SVM model -gradual under-sampling*********************
  under_data_svm=under_data;
  under_weight_svm=under_weight;
  while   F_measure_after(i,2)>=F_measure_before(i,2)   
   if F_measure_after(i,2)==1
       break;
   end    
   if  F_measure_after(i,2)>F_measure_before(i,2)     
        F_measure_before(i,2)=F_measure_after(i,2);  
   end
    
   sorted_weight=under_weight_svm(:,1);
   sorted_data=under_data_svm;
   sort_lable=under_data_svm(:,end);
   
   %under-sampling data*************************
  [under_weight_svm,under_data_svm,five_percent]=data_deletion_func(sorted_weight,sorted_data,sort_lable); 
  if five_percent==0
      break;
  end   
  
   %create SVM model after under-sampling**************************
   model_svm_after = fitcsvm(under_data_svm(:,1:end-1),under_data_svm(:,end),'Standardize',true,'KernelFunction',kernel_svm,'KernelScale','auto');
   predicted_label_svm_after = predict(model_svm_after,data1(test(),:));
   xs_svm_after=confusionmat(lable(test(),:),predicted_label_svm_after);
   [recall_after(i,2),precision_after(i,2),F_measure_after(i,2),G_means_after(i,2),accuracy_after(i,2)]=...
                                                               measures_of_classify(xs_svm_after);
  end
  if   F_measure_after(i,2)<F_measure_before(i,2)
       F_measure_after(i,2)=F_measure_before(i,2);
  end  
%% Draw a graph after under-sampling with SVM model
%     b_data=under_data_svm(:,1:end-1);
%     b_lable=under_data_svm(:,end);
%     figure, scatter(b_data(:,1), b_data(:,2),[], b_lable,'filled');
%     %title('under-sampling dataset');
%     xlabel('x');
%     ylabel('y');
%     colormap([0 0 1;1 0 0])  %RGB for example R=1 0 0

   
   %% while loop for CART model - gradual under-sampling******************
   under_data_cart=under_data;
   under_weight_cart=under_weight;
   while  F_measure_after(i,3)>=F_measure_before(i,3)  
   if F_measure_after(i,3)==1
       break;
   end
   
   if  F_measure_after(i,3)>F_measure_before(i,3)  
       F_measure_before(i,3)=F_measure_after(i,3);  
   end  
   sorted_weight=under_weight_cart(:,1);
   sorted_data=under_data_cart;
   sort_lable=under_data_cart(:,end);
   
   % under-sampling data*************************
   [under_weight_cart,under_data_cart,five_percent]=data_deletion_func(sorted_weight,sorted_data,sort_lable); 
   if five_percent==0
      break;
   end   
  
   % create CART model after under-sampling*******************************
   Model_cart_after = fitctree(under_data_cart(:,1:end-1),under_data_cart(:,end),'MaxNumSplits',MaxNumSplits);
   predicted_label_cart_after = predict(Model_cart_after,data1(test(),:));
   xs_cart_after=confusionmat(lable(test(),:),predicted_label_cart_after);
   [recall_after(i,3),precision_after(i,3),F_measure_after(i,3),G_means_after(i,3),accuracy_after(i,3)]=...
                                                               measures_of_classify(xs_cart_after);
   end
   if  F_measure_after(i,3)<F_measure_before(i,3)
       F_measure_after(i,3)=F_measure_before(i,3);
   end   

%% Draw a graph after under-sampling with CART model
%    b_data=under_data_cart(:,1:end-1);
%    b_lable=under_data_cart(:,end);
%    figure, scatter(b_data(:,1), b_data(:,2),[], b_lable,'filled');
%    %title('under-sampling dataset');
%    xlabel('x');
%    ylabel('y');
%    colormap([0 0 1;1 0 0])  %RGB for example R=1 0 0

%     end_time=toc;
%     disp([ ' Time = '   num2str(end_time) ' seconds' ])
end   
%% Keeping the criteria of each replicate to calculate the standard deviation ****************
%*********************for knn model*****************************
all_recall(:,j,1)=recall_after(:,1);
all_precision(:,j,1)=precision_after(:,1);
all_F_measure(:,j,1)=F_measure_after(:,1);
all_G_means(:,j,1)=G_means_after(:,1);
all_accuracy(:,j,1)=accuracy_after(:,1);
%*********************for SVM model*****************************
all_recall(:,j,2)=recall_after(:,2);
all_precision(:,j,2)=precision_after(:,2);
all_F_measure(:,j,2)=F_measure_after(:,2);
all_G_means(:,j,2)=G_means_after(:,2);
all_accuracy(:,j,2)=accuracy_after(:,2);
%*********************for CART model****************************
all_recall(:,j,3)=recall_after(:,3);
all_precision(:,j,3)=precision_after(:,3);
all_F_measure(:,j,3)=F_measure_after(:,3);
all_G_means(:,j,3)=G_means_after(:,3);
all_accuracy(:,j,3)=accuracy_after(:,3);

end
%% criteria average in 5 repetitions *****************************
% *********************Average_KNN Model*****************************
Ave_recall_knn=sum(sum(all_recall(:,:,1)))/(iter*kf);
Ave_precision_knn=sum(sum(all_precision(:,:,1)))/(iter*kf);
Ave_F_measure_knn=sum(sum(all_F_measure(:,:,1)))/(iter*kf);
Ave_G_means_knn=sum(sum(all_G_means(:,:,1)))/(iter*kf);
Ave_accuracy_knn=sum(sum(all_accuracy(:,:,1)))/(iter*kf);

% *********************Average_SVM Model*****************************
Ave_recall_svm=sum(sum(all_recall(:,:,2)))/(iter*kf);
Ave_precision_svm=sum(sum(all_precision(:,:,2)))/(iter*kf);
Ave_F_measure_svm=sum(sum(all_F_measure(:,:,2)))/(iter*kf);
Ave_G_means_svm=sum(sum(all_G_means(:,:,2)))/(iter*kf);
Ave_accuracy_svm=sum(sum(all_accuracy(:,:,2)))/(iter*kf);

% *********************Average_CART Model****************************
Ave_recall_cart=sum(sum(all_recall(:,:,3)))/(iter*kf);
Ave_precision_cart=sum(sum(all_precision(:,:,3)))/(iter*kf);
Ave_F_measure_cart=sum(sum(all_F_measure(:,:,3)))/(iter*kf);
Ave_G_means_cart=sum(sum(all_G_means(:,:,3)))/(iter*kf);
Ave_accuracy_cart=sum(sum(all_accuracy(:,:,3)))/(iter*kf);

%% calculate std.Dev.***************************
% *********************Std_Dev_KNN Model*****************************
Std_Dev_knn_recall=sqrt(sum(sum((all_recall(:,:,1)-Ave_recall_knn).^2))/(iter*kf));
Std_Dev_knn_precision=sqrt(sum(sum((all_precision(:,:,1)-Ave_precision_knn).^2))/(iter*kf));
Std_Dev_knn_F_measure=sqrt(sum(sum((all_F_measure(:,:,1)-Ave_F_measure_knn).^2))/(iter*kf));
Std_Dev_knn_G_means=sqrt(sum(sum((all_G_means(:,:,1)-Ave_G_means_knn).^2))/(iter*kf));
Std_Dev_knn_accuracy=sqrt(sum(sum((all_accuracy(:,:,1)-Ave_accuracy_knn).^2))/(iter*kf));

% *********************Std_Dev_SVM Model*****************************
Std_Dev_svm_recall=sqrt(sum(sum((all_recall(:,:,2)-Ave_recall_svm).^2))/(iter*kf));
Std_Dev_svm_precision=sqrt(sum(sum((all_precision(:,:,2)-Ave_precision_svm).^2))/(iter*kf));
Std_Dev_svm_F_measure=sqrt(sum(sum((all_F_measure(:,:,2)-Ave_F_measure_svm).^2))/(iter*kf));
Std_Dev_svm_G_means=sqrt(sum(sum((all_G_means(:,:,2)-Ave_G_means_svm).^2))/(iter*kf));
Std_Dev_svm_accuracy=sqrt(sum(sum((all_accuracy(:,:,2)-Ave_accuracy_svm).^2))/(iter*kf));

% *********************Std_Dev_CART Model****************************
Std_Dev_cart_recall=sqrt(sum(sum((all_recall(:,:,3)-Ave_recall_cart).^2))/(iter*kf));
Std_Dev_cart_precision=sqrt(sum(sum((all_precision(:,:,3)-Ave_precision_cart).^2))/(iter*kf));
Std_Dev_cart_F_measure=sqrt(sum(sum((all_F_measure(:,:,3)-Ave_F_measure_cart).^2))/(iter*kf));
Std_Dev_cart_G_means=sqrt(sum(sum((all_G_means(:,:,3)-Ave_G_means_cart).^2))/(iter*kf));
Std_Dev_cart_accuracy=sqrt(sum(sum((all_accuracy(:,:,3)-Ave_accuracy_cart).^2))/(iter*kf));

end