function [recon_test_data, eig_rate] = L1_IRF(tr_data, test_data, comp_consider, n_pc)
%%%%%%%%%%%%%%%%%%%%%%%%%% PCA part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data_pca, w_pca, mean_f, std_f] = PCAL1_wo_writing(tr_data, n_pc);
[N_Tr, temp] = size(tr_data);
N_F = temp-1;
%%%%%%%%%%%%%%%%%%%%%%%%% Reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%
recon_test = [];
for idx_test = 1:size(test_data,1)
    normal_test = data_normalize(test_data(idx_test,1:end-1),mean_f,std_f);
% threshold = 10^(-1);
% [y_next,y_t] = fn_irf_review(test_data(:,1:end-1), w_pca, threshold);    
    recon_data1 = zeros(N_F,1);
    t1 = clock;
    for N_wpca = 1:n_pc
%         y_final = normal_test*w_pca(:,N_wpca);
        y_final = fn_irf_review(normal_test, w_pca(:,N_wpca),0.001);
        recon_data1 = recon_data1 + w_pca(:,N_wpca)*y_final;            
    end
    elap_time = etime(clock, t1);
    display(['IRF running time = ', num2str(elap_time)]);
    recon_data1 = recon_data1.*std_f' + mean_f';
    recon_data = [recon_data1' test_data(idx_test,end)];
    recon_test = [recon_test; recon_data];    
end
recon_test_data = recon_test;
