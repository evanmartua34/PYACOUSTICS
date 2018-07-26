if nargin==1; opt=struct; end
ofld={'f_thresh' 'length' 'rlength' 'fs' 'ret'};
odef={0.0767 0.1524 5 16000 's'};
opt=fu_optstruct_init(opt,ofld,odef);


%%%% preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stereo->mono, mean 0
s = s(:,1)-mean(s(:,1));
% low pass filtering (just carried out if fs > 20kHz)
s = fu_filter(s,'low',10000,opt.fs);


%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reference window span
rws = floor(opt.rlength*opt.fs);
% signal length
ls=length(s);
% min pause length in samples
ml=floor(opt.length*opt.fs);
% global rmse and pause threshold
e_glob = fu_rmse(s);
t_glob = opt.f_thresh*e_glob;
% stepsize
%sts=floor(ml/4);
sts=max(1,floor(0.05*opt.fs));
stsh=floor(sts/2); % for centering of reference window


%%%% pause detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output array collecting pause sample indices
t=[];
j=1;


for i=1:sts:ls
    %%%% window %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yi=i:min(ls,i+ml-1);
    %tt=[yi(1) yi(end)]
    y=s(yi);
    e_y = fu_rmse(y);
    %%%% reference window %%%%%%%%%%%%%%%%%%%
    rw=s(fu_i_window(min(i+stsh,ls),rws,ls));
    e_rw=fu_rmse(rw);
    if (e_rw <= t_glob); e_rw=e_glob; end
    %%%% if rmse in window below threshold %%
    if e_y <= e_rw*opt.f_thresh
        if size(t,1)==j
            % values belong to already detected pause
            if yi(1) < t(j,2)
                t(j,2)=yi(end);
            else                          % new pause
                j=j+1;
                t(j,:)=[yi(1) yi(end)];
            end
        else                              % new pause
            t(j,:)=[yi(1) yi(end)];
        end
    end
end


%%%%%% conversion of sample indices into %%%%%%%%%%%%%%
%%%%%% time on- and offset values (sec) %%%%%%%%%%%%%%%

if strcmp(opt.ret,'s'); t=t./opt.fs; end

return
