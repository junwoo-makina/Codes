%	1DPCA
%	by Spinoz Kim (spinoz@csl.snu.ac.kr)
%	Dec. 8 2004

function [w_pca,elap_time] = L2PCA_new(tr_data, comp_consider, n_sel);

%Read input file
[N_Tr, N_F] = size(tr_data);
%N_F = temp-1;     %N_Tr = 100*2, %N_F = 100*120
data_tr = tr_data;
%class_tr = tr_data(:,N_F+1);
clear tr_data;


%Start the watch for lasting time of the feature extraction
t0 = clock;

%PCA
if comp_consider == 1,
   data_tr_tp = data_tr';
   K = data_tr * data_tr_tp;     %K : (N_Tr) * (N_Tr), Caution that this cov. is correct only if already normailized.
   K = K / N_Tr;
   [q_comp, d] = eig(K);
   for i=1:N_Tr,
      d_temp(i) = d(i,i);
   end
   clear K;  clear d;

   q = data_tr_tp * q_comp;   
   [eig_val, d_idx] = sort(d_temp,'descend');  
   w_pca = zeros(N_F, n_sel);
   for i=1:n_sel,
      w_pca(:,i) = q(:,d_idx(i));   %w_pca : (N_F) * (n_sel), Caution that n_sel should be less than N_Tr in this case.
      norm_weight = norm(w_pca(:,i));
      w_pca(:,i) = w_pca(:,i) / norm_weight;
   end
   clear data_tr_tp;
else,
   data_tr_tp = data_tr';
   K = data_tr_tp * data_tr;     %K : (N_F) * (N_F), Caution that this cov. is correct only if already normailized.
   clear data_tr_tp;
   K = K / N_Tr;
   [q, d] = eig(K);
   for i=1:N_F,
      d_temp(i) = d(i,i);
   end
   
   [eig_val, d_idx] = sort(d_temp,'descend');
   w_pca = zeros(N_F, n_sel);
   for i=1:n_sel,
      w_pca(:,i) = q(:,d_idx(i));   %w_pca : (N_F) * (n_sel)
      norm_weight = norm(w_pca(:,i));
      w_pca(:,i) = w_pca(:,i) / norm_weight;
   end
end

%Finish the stop watch
elap_time = etime(clock, t0);
%display('L2PCA end');
%display(elap_time);

