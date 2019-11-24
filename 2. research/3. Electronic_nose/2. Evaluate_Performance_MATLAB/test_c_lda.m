close all
clc
%%
tr_file_name = 'LDA_150';
test_name = 'ori_e_nose_data.dat';
test_data = load(test_name);

%data parsing
[N_Tst, temp] = size(test_data);
N_F = temp-1;     %N_Tst = 100, %N_F = 100*120
data_test = test_data(:,1:N_F);
class_test = test_data(:,N_F+1);

%Normalize
m_file_name = sprintf('%s_mean.dat', tr_file_name);
m_data = load(m_file_name);
mean_f = m_data(1,:);
std_f = m_data(2,:);
for i=1:N_Tst,
   for j=1:N_F,
      data_test(i,j) = (data_test(i,j)-mean_f(j)) / std_f(j);
      
   end
end

%Data projection
w_file_name = sprintf('%s_weight.dat', tr_file_name);
w_prj = load(w_file_name);
[temp, N_NewF] = size(w_prj);
data_prj = data_test * w_prj;
clear data_test;   