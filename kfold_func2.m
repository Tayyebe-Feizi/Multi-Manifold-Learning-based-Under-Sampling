function [index]=kfold_func2(data,kf)

%Dividing the data into k parts or rewriting Kfold approach***************

[r,c]=size(data);
ratio=floor(r/kf);
mod_d=mod(r,kf);
num_fold=ones(1,kf)*ratio;

for i=1:mod_d
    num_fold(1,i)=num_fold(1,i)+1;
end

start_index=1;
end_index=num_fold(1,1);

for  j=1:kf
      data(start_index:end_index,c+1)=j;
      if j<kf
      start_index=end_index+1;
      end_index=start_index+num_fold(1,j+1)-1;
      end   
end
index=data(:,end);
end