function [under_weight,under_data,five_percent]=data_deletion_func(sorted_weight,sorted_data,sort_lable)
%% Initialization of variables
[~,c]=size(sorted_data); 
minority_data=sorted_data(sorted_data(:,c)==1,:);
majority_data=sorted_data(sorted_data(:,c)==2,:);
r_maj=size(majority_data,1);
r_min=size(minority_data,1);

sorted_weight_with_lables=[sorted_weight,sort_lable];   
minority_sorted_weight=sorted_weight_with_lables(sorted_weight_with_lables(:,2)==1,:);
majority_sorted_weight=sorted_weight_with_lables(sorted_weight_with_lables(:,2)==2,:);
%% Removing 5% of the majority class
[r,~]=size(majority_data);
five_percent=round(r*0.05);   % 5 per= 0.05      
majority_sorted_weight(1:five_percent,3)=5;

majority_sorted_weight_decreased=majority_sorted_weight(majority_sorted_weight(:,3)~=5,:);
majority_data_decreased=majority_data(majority_sorted_weight(:,3)~=5,:);

under_weight=[majority_sorted_weight_decreased(:,1:2);minority_sorted_weight];
under_data=[majority_data_decreased;minority_data];

end