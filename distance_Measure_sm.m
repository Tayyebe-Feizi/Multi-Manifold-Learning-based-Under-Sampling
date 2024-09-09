function  [manifold,all_data_map]=distance_Measure_sm(data)
% number of each classes
labels=data(:,end);
class=unique(labels);
%% initialization
manifold=struct;          %name and alpha value of manifold on both of class
all_data_map=struct;      %map on both of class Simultaneously

u_InfoLost=zeros(1,1);
u_alpha=zeros(1,1);

[~,c]=size(data); 
disp('original dimensions for data is : ');
disp(c-1);
original_dims=c-1;
no_dims=original_dims;
%% unsupervised Methods
%Linear  Methods
%PCA:  Principal Component Analysis 
%LPP:  Locality Preserving Projection
%NPE: Neighborhood Preserving Embedding

%% map all data together or  map on both of class Simultaneously
all_X=data(:,1:end-1);

%Select the manifold and execute only one of the lines 26 to 28.
 type='PCA';
% type='LPP';
% type='NPE';

 [mappedX, mapping] = compute_mapping(all_X,type, no_dims);
 all_data_map(1).all_x=mappedX;

  all_data_map(1).all_x=real(all_data_map(1).all_x);    
  all_data_map(1).all_x(isnan(all_data_map(1).all_x))=0;

%% map every class individually or map on only one class individually for single manifold
for i=1:numel(class)
     classNo(i)=numel(find(labels==class(i)));
     classes=data(data(:,c)==i,:);
     nc=classNo(i);
     X =classes(:,1:c-1);      
     label_class=classes(:,c); 
     
%% ***************Mapping with the single manifold****************************
     [mappedX, mapping] = compute_mapping(X,type, no_dims);
     c_m=size(mappedX,2);
     c_x=size(X,2);
     mappedX=real(mappedX);
     mappedX(isnan(mappedX))=0;
     if      c_m < c_x                    % mapping with reduced dimension
             X_hat=mappedX*mapping.M';    
             u= X- X_hat;
     elseif     no_dims == original_dims  % only mapping
             u=X-mappedX;
     end
     u_InfoLost(1,1)=norm(u,2)^2/nc;   %Calculation of information loss  for single manifold
     u_alpha(1,1)=1/u_InfoLost(1,1);   %Calculation of Alpha for single manifold
     u_alpha(isnan(u_alpha))=0;

%% save name and alpha value of manifold on both of class
    manifold(i).alpha=u_alpha;     
    manifold(i).type=type;
      
end
end