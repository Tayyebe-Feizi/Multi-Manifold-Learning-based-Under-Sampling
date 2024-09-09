function  [Degree]=cen_mar_func(data,k)
%The function of calculating the degree of centrality and marginality of each data
   
   [r,~]=size(data); 
 for i=1:r
      sample=data(i,1:end-1);
      lable=data(i,end);
     
     if    lable==1
            other=2;
     else 
            other=1;
     end    
     
     [Idx1,D] = knnsearch(data(:,1:end-1),sample,'K',k, 'Distance','euclidean');
     Idx=Idx1(:,2:end);
 
      num_similarity=numel(find(data(Idx,end)==lable));
      num_opposites=numel(find(data(Idx,end)==other));
  
      Degree(i,1)=num_similarity;
      Degree(i,2)=num_opposites;
  
 end

end