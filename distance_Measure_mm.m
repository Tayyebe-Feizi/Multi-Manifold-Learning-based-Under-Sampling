function  [manifold,all_data_map]=distance_Measure_mm(data)
% number of each classes
labels=data(:,end);
class=unique(labels);
%% initialization
manifold=struct;          %names and alpha values of all manifolds on both of class
all_data_map=struct;      %map on both of class Simultaneously

u_InfoLost=zeros(3,1);
u_alpha=zeros(3,1);

[~,c]=size(data); 
disp('original dimensions for data is : ');
disp(c-1);
original_dims=c-1;
no_dims=original_dims;

%% unsupervised Methods
% Linear  Methods
%PCA:  Principal Component Analysis 
%LPP:  Locality Preserving Projection
%NPE: Neighborhood Preserving Embedding

%% map all data together or  map on both of class Simultaneously
all_X=data(:,1:end-1);
%***************Mapping with the PCA manifold****************************
 type='PCA';
 [mappedX, mapping] = compute_mapping(all_X,type, no_dims);
 all_data_map(1).all_x=mappedX;

%***************Mapping with the LPP manifold****************************
type='LPP';
[mappedX, mapping] = compute_mapping(all_X,type, no_dims);	
all_data_map(2).all_x=mappedX;

%***************Mapping with the NPE manifold****************************
 type='NPE';
 [mappedX, mapping] = compute_mapping(all_X,type, no_dims);	
 all_data_map(3).all_x=mappedX;

 for i=1:3
    all_data_map(i).all_x=real(all_data_map(i).all_x);    
    all_data_map(i).all_x(isnan(all_data_map(i).all_x))=0;
end

%% map every class individually or map on only one class individually for all manifolds
for i=1:numel(class)
     classNo(i)=numel(find(labels==class(i)));
     classes=data(data(:,c)==i,:);
     nc=classNo(i);
     X =classes(:,1:c-1);      
     label_class=classes(:,c); 
     
 %% ***************Mapping with the PCA manifold****************************
     utype_m='PCA';
     type=utype_m;
     [mappedX, mapping] = compute_mapping(X,type, no_dims);
     c_m=size(mappedX,2);
     c_x=size(X,2);
     mappedX=real(mappedX);
     mappedX(isnan(mappedX))=0;

     if      c_m < c_x                      % mapping with reduced dimension
             X_hat=mappedX*mapping.M';      
             u= X- X_hat;
     elseif     no_dims == original_dims    % only mapping
             u=X-mappedX;
     end
     u_InfoLost(1,1)=norm(u,2)^2/nc;         %Calculation of information loss  for PCA manifold
     u_alpha(1,1)=1/u_InfoLost(1,1);         %Calculation of Alpha for PCA manifold
 %% ***************Mapping with the LPP manifold****************************
     type='LPP';
     utype_m=char(utype_m,type);
     [mappedX, mapping] = compute_mapping(X,type, no_dims);	
     
        if   no_dims < original_dims      % mapping with reduced dimension
             X_hat=mappedX*mapping.M';
             u= X- X_hat;
        elseif     no_dims == original_dims       % only mapping
             u=X-mappedX;
       end
       u_InfoLost(2,1)=norm(u,2)^2/nc;      %Calculation of information loss  for LPP manifold
       u_alpha(2,1)=1/u_InfoLost(2,1);      %Calculation of Alpha for LPP manifold

 %% ***************Mapping with the NPE manifold ****************************       
     type='NPE';
     utype_m=char(utype_m,type);
     [mappedX, mapping] = compute_mapping(X,type, no_dims);	
     
        if   no_dims < original_dims      % mapping with reduced dimension
             X_hat=mappedX*mapping.M';
             u= X- X_hat;
        elseif     no_dims == original_dims       % only mapping
             u=X-mappedX;
       end
       u_InfoLost(3,1)=norm(u,2)^2/nc;           %Calculation of information loss for NPE manifold
       u_alpha(3,1)=1/u_InfoLost(3,1);           %Calculation of Alpha for NPE manifold
       u_alpha(isnan(u_alpha))=0;
%% save names and alpha values all manifolds on both of class
    manifold(i).alpha=u_alpha;      
    manifold(i).type=utype_m;
    
end

end