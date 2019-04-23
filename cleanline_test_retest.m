% Example code to load BDF files from the ft10 project,
% filter the data, and illustrate the ERPs.
% Also demonstrates how to add behavioural data to the EEG structure

% for the code to work, you need to:
% - unzip the 4 eeg folders into a new `ft10_eeg` folder
%       `ft10_p1.zip` `ft10_p2.zip` `ft10_p3.zip` `ft10_p4.zip`
% - unzip the folder `ft10_behaviour`
% - cd to the folder containing the two folders above
% - add the file `Glasgow_BioSemi_132.ced` to that folder
% - add to your path / place in the eeglab plugin folder:
%       Arnaud Delorme's BDFimport plugin & ERPLAB
%   both are available at:
%       <http://sccn.ucsd.edu/wiki/EEGLAB_Extensions_and_plug-ins>
% - add to your path the `findgap` function

Np = 4; % number of participants
Ns = 10; % number of sessions
eeg_folder = '/data1/users/adoyle/eeg_test_retest/';
exptname = 'ft10';


for P = 1:Np % for each participant
    for ses = 1:Ns % for each session
        % load data and set reference to the average of electrodes [1:128]
        id = [exptname,'_p',num2str(P),'s',num2str(ses)];
        bdffile = [eeg_folder,'raw/',exptname,'_p',num2str(P),'/',id,'.bdf'];
        %EEG = pop_biosig(bdffile);
        EEG = pop_readbdf(bdffile,[],137,[],'off');
        %EEG = pop_readbdf(bdffile);
        
        %EEG = eeg_checkset( EEG );
        %EEG = pop_biosig(bdffile,[],137,1:128);
        
        %EEG = eeg_checkset( EEG );
        %EEG = pop_chanedit(EEG,  'lookup',[eeg_folder,'Glasgow_BioSemi_132.ced']);
        
        %EEG = pop_select( EEG, 'channel', 1:132);
        %EEG = eeg_checkset( EEG );
        
        EEG = pop_cleanline(EEG, 'bandwidth',3,'chanlist',[1:132] ,'computepower',0,'linefreqs',[50:50:250] ,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype','Channels','tau',100,'verb',1,'winsize',5,'winstep',2.5);
        %EEG = eeg_checkset( EEG );
        
        %EEG = pop_chanevent(EEG, 137,'edge','leading','edgelen',0,'delchan','on','delevent','off');
        %EEG = eeg_checkset( EEG );
           
        out_folder = [eeg_folder,'cleaned/',exptname,'_p',num2str(P),'/'];
        %pop_writeeeg(EEG, [out_folder,id,'.bdf'], 'TYPE','BDF');
        EEG = pop_saveset( EEG, 'filename',[id,'.set'], 'filepath',out_folder);
    end
end