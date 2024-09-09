function  [sorted_weight,sorted_data,sort_lable]=single_manifold(data)
% Calculate the weighted centrality and weighted marginality in a single-manifold approach
[r,~]=size(data); 
labels=data(:,end);

%% Mapping data with the distance criterion
[manifold,all_data_map]=distance_Measure_sm(data);

%% Selection of neighborhoods based on the weighted combination of centrality and marginality in the single-manifold approach

k=5;     % number of neighborhoods for calculate centrality and marginality  
k=k+1;
Degree=zeros(r,2,1);

map_data=[all_data_map(1,1).all_x,labels];
Degree(:,:,1)=cen_mar_func(map_data,k);    % third index for Degree is volume


 weighted_cen=zeros(r,1);
 weighted_mar=zeros(r,1);
 
for i=1:r
   if  map_data(i,end)==1    %for positive data or minitory data
  
   weighted_cen(i,1)=manifold(1,1).alpha(1,1)* Degree(i,1,1);   %Calculate the weighted centrality
   weighted_mar(i,1)=manifold(1,1).alpha(1,1)* Degree(i,2,1);   %Calculate the weighted marginality
                           
   else                       %for negetive data or majority data

    weighted_cen(i,1)=manifold(1,2).alpha(1,1)* Degree(i,1,1);  %Calculate the weighted centrality                  
    weighted_mar(i,1)=manifold(1,2).alpha(1,1)* Degree(i,2,1);  %Calculate the weighted marginality
   end        
end
 weight= weighted_mar-weighted_cen;    %Linear combination of centrality and marginality
[sorted_weight,index]=sort(weight,'descend');
sorted_data=data(index,:);
sort_lable=labels(index);
     
end
