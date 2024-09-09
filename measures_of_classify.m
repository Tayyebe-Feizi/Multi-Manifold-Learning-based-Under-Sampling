function [recall,precision,F_measure,G_means,accuracy]=measures_of_classify(xs)
   
   %Performance measurement criteria
   
   recall=xs(1,1)/sum(xs(1,:));
   recall(isnan( recall))=0;
   precision=xs(1,1)/sum(xs(:,1));
   precision(isnan(precision))=0;
   F_measure=2*(recall*precision)/(recall+precision);
   F_measure(isnan(F_measure))=0;
   Sensitivity_before=recall;
   Specificity_before=xs(2,2)/sum(xs(2,:));
   Specificity_before(isnan(Specificity_before))=0;
   G_means=sqrt(Specificity_before*Sensitivity_before);
   G_means(isnan(G_means))=0;
   accuracy=(xs(1,1)+xs(2,2))/sum(sum(xs));
end