function [tr_prj, te_prj, w_final] = CLDA2(tr_data, te_data, label_tr, n_pc)

%	2DLDA
%	by Spinoz Kim (spinoz@csl.snu.ac.kr)
%	Oct 14 2004

[N_Tr, temp] = size(tr_data);
[N_Te, ~] = size(te_data);
N_F = temp-1;     %N_Tr = 994*3, %N_F = 100*120
data_tr = tr_data(:,1:N_F);
data_te = te_data(:,1:N_F);
%class_tr = tr_data(:,N_F+1);
clear tr_data;

F_Width = 100;
x_overlap = 1;
y_overlap = 1;
comp_x_size = 10;
comp_y_size = 32;
n_midsel = 50;
n_sel = n_midsel;
N_C = 8;

%1-D feature to 2-D feature & image extraction - convert vector to image
for i=1:N_Tr,
   for j=1:N_F,
      x_val = ceil(j/F_Width);
      y_val = mod(j,F_Width);
      if y_val == 0,
         y_val = F_Width;
      end
      tr_image(x_val,y_val,i) = data_tr(i,j);
   end
end
clear data_tr;

for i=1:N_Te,
   for j=1:N_F,
      x_val = ceil(j/F_Width);
      y_val = mod(j,F_Width);
      if y_val == 0,
         y_val = F_Width;
      end
      te_image(x_val,y_val,i) = data_te(i,j);
   end
end
clear data_te;

%code for type3 data immediately, comment is made when comp_x_size=10 and comp_y_size=12
[N_Row, N_Col, N_Tr] = size(tr_image);    %N_Row = 120, N_Col = 100, N_Tr = 2982

clear data_te

Window_Size = comp_y_size * comp_x_size;  %Window_Size = 120

N_FC = (N_Row * N_Col) / Window_Size;  %N_FC = 100
step_x = comp_x_size;   step_y = comp_y_size;
% if bitor(x_overlap, y_overlap) == 0,
%    N_FC = (N_Row * N_Col) / Window_Size;  %N_FC = 100
%    step_x = comp_x_size;   step_y = comp_y_size;
% elseif bitand(x_overlap, y_overlap) == 1,
%    N_FC = (2*(N_Row/comp_y_size)-1) * (2*(N_Col/comp_x_size)-1);
%    step_x = comp_x_size/2;    step_y = comp_y_size/2;
% elseif x_overlap == 1,
%    N_FC = (N_Row/comp_y_size) * (2*(N_Col/comp_x_size)-1);
%    step_x = comp_x_size/2;    step_y = comp_y_size;
% elseif y_overlap == 1,
%    N_FC = (2*(N_Row/comp_y_size)-1) * (N_Col/comp_x_size);
%    step_x = comp_x_size;    step_y = comp_y_size/2;
% end

for i=1:N_Tr,
   window_idx = 0;
   for j=1:step_y:(N_Row-comp_y_size+1),
      for k=1:step_x:(N_Col-comp_x_size+1),
         target = tr_image(j:(j+comp_y_size-1), k:(k+comp_x_size-1), i);
         target_vec = reshape(target', 1, Window_Size);     %size(target_vec) = [1, Window_Size]
         window_idx = window_idx+1;
         tr_type3(:,window_idx,i) = target_vec';
      end
   end
end
clear tr_image;

for i=1:N_Te,
   window_idx = 0;
   for j=1:step_y:(N_Row-comp_y_size+1),
      for k=1:step_x:(N_Col-comp_x_size+1),
         target = te_image(j:(j+comp_y_size-1), k:(k+comp_x_size-1), i);
         target_vec = reshape(target', 1, Window_Size);     %size(target_vec) = [1, Window_Size]
         window_idx = window_idx+1;
         te_type3(:,window_idx,i) = target_vec';
      end
   end
end
clear te_image;

%Normalize
tr_type3 = double(tr_type3);
te_type3 = double(te_type3);
[N_Row, N_Col, N_Tr] = size(tr_type3);   
[N_Row, N_Col, N_Te] = size(te_type3); 
[mean_mat, std_mat] = cal_std_mat(tr_type3);
[mean_mate, std_mate] = cal_std_mat(te_type3);
%error check
std_mat(find(std_mat== 0)) = 1;
std_mate(find(std_mate== 0)) = 1;

for i=1:N_Tr,
   for j=1:N_Row,
      for k=1:N_Col,
         tr_type3(j,k,i) = (tr_type3(j,k,i) - mean_mat(j,k)) / std_mat(j,k);
      end
   end
end

for i=1:N_Te,
   for j=1:N_Row,
      for k=1:N_Col,
         te_type3(j,k,i) = (te_type3(j,k,i) - mean_mate(j,k)) / std_mate(j,k);
      end
   end
end
% fid = fopen([out_file,'_mean.dat'], 'w');
% for i=1:N_Row,
%    for j=1:N_Col,
%       fprintf(fid,'%.4f ', mean_mat(i,j));
%    end
%    fprintf(fid,'\n');
% end
% for i=1:N_Row,
%    for j=1:N_Col,
%       fprintf(fid,'%.4f ', std_mat(i,j));
%    end
%    fprintf(fid,'\n');
% end
% fclose(fid);
%display('Normalization end');

%Start the watch for lasting time of the feature extraction
t0 = clock;

%Image Within-cov. matrix, Cw calculating
for i=1:N_C,
   IDX = find(label_tr == i);
   target = tr_type3(:,:,IDX);  %Caution! Class label should be sorted and has the same number of samples.
   mean_image(:,:,i) = cal_mean_mat(target);
end

Cw = zeros(N_Col, N_Col);
for i=1:N_Tr,
   class_idx = label_tr(i,1);
   data_rest = tr_type3(:,:,i) - mean_image(:,:,class_idx);
   Cw = Cw + (data_rest'*data_rest);
end
Cw = Cw / N_Tr;
%Cw = Cw / (N_Tr-1);
ra_Cw = rank(Cw);   %display(ra_Cw);

%Image Between-cov. matrix, Cb calculating
mean_tot = cal_mean_mat(tr_type3);

Cb = zeros(N_Col, N_Col);
for i=1:N_C,
   data_rest = mean_image(:,:,i) - mean_tot;
   Cb = Cb + (data_rest'*data_rest);
end
Cb = Cb / N_C;    %Caution! # of samples per class is same.
%Cb = Cb / (N_C-1);
ra_Cb = rank(Cb);   %display(ra_Cb);
clear data_rest;  clear mean_tot;   clear mean_image;

%Eigenvalue decomposition of Cw
[q_1tot, d_1tot] = eig(Cw);
for i=1:N_Col,
   d1_value(i) = d_1tot(i,i);
end
[Cweig_val, d_idx] = sort(d1_value);   %sort in ascending order

for i=1:n_midsel,
   q_1(:,i) = q_1tot(:,d_idx(N_Col-i+1));
   d_1(i) = Cweig_val(N_Col-i+1);
end

d_invroot = zeros(n_midsel,n_midsel);
for i=1:n_midsel,
   d_invroot(i,i) = sqrt(1/d_1(i));
end
w_1 = q_1 * d_invroot;     %dim(w_1) : N_Col * n_midsel

Cw_1 = w_1' * Cw * w_1;    %Cw_1 : Identity matrix
Cb_1 = w_1' * Cb * w_1;

%Eigenvalue decomposition of Cb_1
[q_2, d_2] = eig(Cb_1);    %Cb_1 : N_Col * N_Col

for i=1:n_midsel,
   d2_value(i) = d_2(i,i);
end
[Cbeig_val, d_idx] = sort(d2_value);   %sort in ascending order

for i=1:n_sel,
   w_2(:,i) = q_2(:,d_idx(n_midsel-i+1));
end

%Finish the stop watch
elap_time = etime(clock, t0);
%display('2DLDA end');
%display(elap_time);

% fid = fopen([out_file,'_ldaeig.dat'], 'w');
% fprintf(fid,'<Cw eigenvalues>\n');
% for i=1:n_midsel,
%    fprintf(fid,'%.4f ', Cweig_val(end-i+1));
% end
% fprintf(fid,'\n\n');
eig_tot = sum(Cweig_val);
for i=10:10:n_midsel,
   eig_extracted = sum(Cweig_val(end-i+1:end));
   eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
%    fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));   
end
% fprintf(fid,'\n\n');
% fprintf(fid,'<Cb_1 eigenvalues>\n');
% for i=1:n_sel,
%    fprintf(fid,'%.4f ', Cbeig_val(end-i+1));
% end
% fprintf(fid,'\n\n');
eig_tot = sum(Cbeig_val);
for i=10:10:n_sel,
   eig_extracted = sum(Cbeig_val(end-i+1:end));
   eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
%    fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));   
end
% fprintf(fid,'\n\n');
% fprintf(fid,'elap_time : %.2f (sec)\n', elap_time);
% fclose(fid);

w_final = w_1 * w_2;
% [N_FC, N_Weight] = size(w_final);
% % fid = fopen([out_file,'_weight.dat'], 'w');
% % for i=1:N_FC,
% %    for j=1:N_Weight,
% %       fprintf(fid,'%.4f ', w_final(i,j));
% %    end
% %    fprintf(fid,'\n');
% % end
% % fclose(fid);
% 
%Cov. matrix calculation
%if metric_flag == 3,    %Mahalanobis metric
%   Cw_2 = w_2' * Cw_1 * w_2;    %Cw_2 : Identity matrix
%   Cb_2 = w_2' * Cb_1 * w_2;    %Cb_2 : Diagonal matrix
%   cov_prj = Cw_2 + Cb_2;    %Due to "C = Cw + Cb"
%   clear Cw_2;    clear Cb_2;
%else,
%   cov_prj = 0;
%end
% 
%Tr. data projection
compression_x = 5;
compression_y = 16;
N_Weight = 50;

if (compression_x == 1) & (compression_y == 1),
   tr_prj = zeros(N_Row, N_Weight, N_Tr);
   for i=1:N_Tr,
      target = tr_type3(:,:,i) * w_final;
      tr_prj(:,:,i) = target;
   end
%    
else,
   block_size = compression_y*compression_x;
   mul = zeros(N_Row/block_size, N_Col);
   tr_prj = zeros(N_Row/block_size, N_Weight, N_Tr);
   te_prj = zeros(N_Row/block_size, N_Weight, N_Te);
   for i=1:N_Tr,
      for j=1:N_Col,
         img = (reshape(tr_type3(:,j,i), comp_x_size, comp_y_size))';   %size(img) = [comp_y_size, comp_x_size]      
         inc = 0;
         for idx_y=1:compression_y:comp_y_size,
            for idx_x=1:compression_x:comp_x_size,
               inc=inc+1;
               block = img(idx_y:(idx_y+compression_y-1), idx_x:(idx_x+compression_x-1));
               block_vec = reshape(block, block_size, 1);
               img_compressed(inc,1) = mean(block_vec);
            end
         end
      
         mul(:,j) = img_compressed;
      end
      tr_prj(:,:,i) = mul * w_final;
   end
   for i=1:N_Te,
      for j=1:N_Col,
         img = (reshape(te_type3(:,j,i), comp_x_size, comp_y_size))';   %size(img) = [comp_y_size, comp_x_size]      
         inc = 0;
         for idx_y=1:compression_y:comp_y_size,
            for idx_x=1:compression_x:comp_x_size,
               inc=inc+1;
               block = img(idx_y:(idx_y+compression_y-1), idx_x:(idx_x+compression_x-1));
               block_vec = reshape(block, block_size, 1);
               img_compressed(inc,1) = mean(block_vec);
            end
         end
      
         mul(:,j) = img_compressed;
      end
      te_prj(:,:,i) = mul * w_final;
   end
end
clear tr_type3;
clear te_type3;


end