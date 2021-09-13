%%%%%% motion simulation
%
% 2020.12.01
% Li Zhenghao

clear;
close all;
clc
current_path = pwd;

%% path
t1_path = '\\192.168.1.2\nas\lzh_znso4\interventional\origin';
m_path = '\\192.168.1.2\nas\lzh_znso4\interventional\mask';
slice_path = '\\192.168.1.2\nas\lzh_znso4\interventional\silce';

%% motion simulation
% get t1 data and corresponding mask files
[t1_num,t1_name] = GetFiles(t1_path); 
[m_num,m_name] = GetFiles(m_path); 

num = 5;  % each slice will be added motion n times, and get n motion slice

% add motion
for i = 1:t1_num
    cd(slice_path);   
    mkdir(t1_name{i});cd(t1_name{i});   % create and go to NO.i data folder
    
    % read
    [t1_nii,t1_volume] = nii_read_spm([t1_path,'\',t1_name{i}]);    % read NO.i t1 data
    [m_nii,m_volume] = nii_read_spm([m_path,'\',m_name{i}]);        % read NO.i mask
    
    % cut slices
    v_size = size(t1_volume);
    
    for j = 1:v_size(3)
        if max(max(m_volume(:,:,j))) ~= 0
            slice = t1_volume(:,:,j);
            slice = flipud(slice');     % change direction
            r_pad = floor((195-v_size(1))/2);
            l_pad = ceil((195-v_size(1))/2);
            slice = [zeros(v_size(2),l_pad),slice,zeros(v_size(2),r_pad)];
            for k = 1:num
                % add motion
                post_slice = add_motion(slice);
                subj = t1_name{i}(1:end-4);         % NO.i data
                NO_slice = num2str(j);              % NO.j slice
                motion_time = num2str(k);           % NO.k motion added
                post_slice = abs(post_slice);
                save([subj,'_',NO_slice,'_',motion_time,'.mat'],'post_slice');
%                 figure,imshow(post_slice,[]);
            end
            
            % save label
            save([subj,'_label_',NO_slice,'.mat'],'slice');
%             figure,imshow(slice,[]);
            imwrite(mat2gray(slice),[subj,'_label_',NO_slice,'.tif']);
            
        end
    end
    
    % save max and min of t1 data
    max_t1 = max(max(max(t1_volume)));
    min_t1 = min(min(min(t1_volume)));
    max_min = [max_t1,min_t1];
    save('max_min.mat','max_min');
    
    fprintf('%d/%d done.\n',i,t1_num);
end

%% end
cd(current_path);

%% function add motion
function post_slice = add_motion(slice)
% displacement and rotation, output same size of silce 
psize = 1;          % pixel size
n = 2;              % rotation times
var_angle = 2;      % max abs angle
score_total = 0;    % score

size_t1 = size(slice);      % slice size

while 1
    for i = 1:n
        % rotation
        angle = normrnd(0,var_angle,1,1);
        slice_temp1 = imrotate(slice,-angle,'nearest','crop');

        % displacement
        dx = normrnd(angle,0.1,1,1);
        slice_temp2 = zeros(size_t1(1),size_t1(2));
        if round(dx/psize) > 0
            slice_temp2(:,round(dx/psize):end) = slice_temp1(:,1:end-round(dx/psize)+1);
        else
            slice_temp2(:,1:end+round(dx/psize)) = slice_temp1(:,-round(dx/psize)+1:end);
        end

        % fft
        fft2_temp = fft2(slice_temp2);
        final_fft( (i-1)*(size_t1(1)/n)+1:i*(size_t1(1)/n),:) = ...
            fft2_temp( (i-1)*(size_t1(1)/n)+1:i*(size_t1(1)/n),:);

        % score
        dR = 64 * sqrt(2-2*cos(angle/180*pi));
        score_i = dR + dx^2;
        score_total = score_total + score_i^2;
    end
    
    score_total = sqrt(score_total);
    
    % score_total need to be greater than 5, 
    % then it will be considered has valid a motion
    if score_total > 5
        post_slice = ifft2(final_fft);
        break;
    end
end
end




