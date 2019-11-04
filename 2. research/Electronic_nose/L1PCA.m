%	1DPCA
%	by Spinoz Kim (spinoz@csl.snu.ac.kr)
%	Dec. 8 2004

function [w, n_it, elap_time] = L1PCA(tr_data, Ns);

%Read input file
%[N, temp] = size(tr_data);
[N, N_f] = size(tr_data);
%N_f = temp-1;     %N_Tr = 100*2, %N_F = 100*120
data_tr = tr_data(:,1:N_f);
%class_tr = tr_data(:,N_f+1);
%clear tr_data;


if N < N_f % number of samples is smaller than dimension
   comp_consider = 1;
   r = rank(data_tr);
   [w_pca, temp_time] = L2PCA_new(tr_data, comp_consider, r);
   x = tr_data * w_pca;   
else 
   comp_consider = 0;
   x = data_tr;
   r = N_f;
end

%Start the watch for lasting time of the feature extraction
t0 = clock;

%% from here: searching direction
%x = data_tr;
w = [];
for i=1:Ns
    if i~=1
        x = x - (x*w(:,i-1))*w(:,i-1)';  % x is a row vector, v is a column vector
    end
    
    n = 0;
    % find maximum x_i
    for j=1:N
        norm_x(j) = norm(x(j,:));
    end
    
    [sorted_norm_x, ind] = sort(norm_x);
    v = x(ind(end),:)' / sorted_norm_x(end);
    
    %penalize
    med_norm = median(norm_x);
    for j=1:N
        if norm_x(j) > med_norm
            penal(j) = (med_norm/norm_x(j));
        else
            penal(j) = 1;
        end
    end
    
    %% random direction
    %index = ceil(rand(1)*N);
    %v = x(index,:)' / norm(x(index,:));
    
    %%initialize by PCA
    %v = L2PCA(x, 1, 1);
    
    v_old = zeros(r,1); % initial direction    
    while ((v ~= v_old))
        v_old = v;
        % check polarity of inner product
        sum_x = zeros(1,r);
        for j=1:N
            if x(j,:) * v >= 0 
                p(j) = 1;
            else
                p(j) = -1;
            end
            sum_x = sum_x + penal(j)*p(j)*x(j,:);
        end
        %abs_sum_x = sqrt(sum(sum_x.^2));
        v = sum_x'/norm(sum_x);%abs_sum_x;
        n = n+1;
    end
    w = [w, v];  
    n_it(i) = n;
end

if comp_consider == 1,
    w = w_pca*w;
end

%Finish the stop watch
elap_time = etime(clock, t0);
%display('L1PCA end');
%display(elap_time);

%% projection

%fid = fopen([out_file,'_pcaeig.dat'], 'w');
%for i=1:n_sel,
%   fprintf(fid,'%.4f ', eig_val(i));
%end
%fprintf(fid,'\n\n');
%eig_tot = sum(eig_val);
%for i=1:n_sel,
%   eig_extracted = sum(eig_val(1:i));
%   eig_rate_vec(i) = (eig_extracted/eig_tot) * 100;
%   fprintf(fid,'eig_rate(%d features) : %.2f\n', i, eig_rate_vec(i));   
%end
%fprintf(fid,'\n\n');
%fprintf(fid,'elap_time : %.2f (sec)\n', elap_time);
%fclose(fid);

%Tr. data projection
%tr_prj = data_tr * w_pca;
clear data_tr;

%res = 1;
