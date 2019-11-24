function [w_final,elapse_time] = CLDA(data_tr, label_tr, n_pc)

t_start = clock;
%Define parameters
%tr_file_name = 'Feret_Tr200.dat';
Tr_Each = 1;
comp_consider = 1;   %Cov. matrix : N_TrxN_Tr, in case of comp_consider=1
%n_pc = 150;          %n_pc <= N_Tr (=200), in case of comp_consider=1
%n_midsel = 150;      %n_midsel <= N_Tr-N_C (=100)
%n_sel = 150;         %n_sel <= N_C-1 (=99)
%out_file = 'LDA_150';


%do clda
%data_tr = tr_data(:,1:N_F);
%class_tr = label_tr;
[N_Tr, N_F] = size(data_tr);
%N_F = 320;
N_C = N_Tr;
n_midsel = 10;
n_sel = n_midsel;

%Normalize
[mean_f, std_f] = cal_std(data_tr);
for i=1:N_Tr,
   for j=1:N_F,
      data_tr(i,j) = (data_tr(i,j)-mean_f(j)) / std_f(j);
   end
end

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
   [eig_val, d_idx] = sort(d_temp);    %sort in ascending order
   w_pca = zeros(N_F, n_pc);
   for i=1:n_pc,
      w_pca(:,i) = q(:,d_idx(N_Tr-i+1));   %w_pca : (N_F) * (n_pc), Caution that n_pc should be less than N_Tr in this case.
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
   
   [eig_val, d_idx] = sort(d_temp);
   w_pca = zeros(N_F, n_pc);
   for i=1:n_pc,
      w_pca(:,i) = q(:,d_idx(N_F-i+1));   %w_pca : (N_F) * (n_pc)
   end
end

data_pca = data_tr * w_pca;

%Within-cov. matrix, Cw calculating
for i=1:N_C,
   mean_image(i,:) = mean(data_pca(((i-1)*Tr_Each+1):(i*Tr_Each), :));  %Caution! Class label should be sorted and has the same number of samples.
end

Cw = zeros(n_pc, n_pc);
for i=1:N_Tr,
   class_idx = floor(((i-1)/Tr_Each) + 1);
   data_rest = data_pca(i,:) - mean_image(class_idx,:);
   Cw = Cw + (data_rest'*data_rest);
end
Cw = Cw / N_Tr;
%Cw = Cw / (N_Tr-1);
ra_Cw = rank(Cw);   display(ra_Cw);

%Between-cov. matrix, Cb calculating
mean_tot = mean(data_pca);

Cb = zeros(n_pc, n_pc);
for i=1:N_C,
   data_rest = mean_image(i,:) - mean_tot;
   Cb = Cb + (data_rest'*data_rest);
end
Cb = Cb / N_C;    %Caution! # of samples per class is same.
%Cb = Cb / (N_C-1);
ra_Cb = rank(Cb);   display(ra_Cb);
clear data_rest;  clear mean_tot;   clear mean_image;

%Eigenvalue decomposition of Cw
[q_1tot, d_1tot] = eig(Cw+Cw');
for i=1:n_pc,
   d1_value(i) = d_1tot(i,i);
end
[Cweig_val, d_idx] = sort(d1_value);   %sort in ascending order

for i=1:n_midsel,
   q_1(:,i) = q_1tot(:,d_idx(n_pc-i+1));
   d_1(i) = Cweig_val(n_pc-i+1);
end

d_invroot = zeros(n_midsel,n_midsel);
for i=1:n_midsel,
   d_invroot(i,i) = sqrt(1/abs(d_1(i)));
end
w_1 = q_1 * d_invroot;     %dim(w_1) : n_pc * n_midsel

Cw_1 = w_1' * Cw * w_1;    %Cw_1 : Identity matrix
Cb_1 = w_1' * Cb * w_1;

%Eigenvalue decomposition of Cb_1
[q_2, d_2] = eig(Cb_1'+Cb_1);

for i=1:n_midsel,
   d2_value(i) = d_2(i,i);
end
[Cbeig_val, d_idx] = sort(d2_value);   %sort in ascending order

for i=1:n_sel,
   w_2(:,i) = q_2(:,d_idx(n_midsel-i+1));
end


w_lda = w_1 * w_2;
w_final = w_pca * w_lda;

%Finish the stop watch
elapse_time = etime(clock, t_start);

end