% /*----------------------------------------------------------------------
%     This file contains a simulation of the cart and pole dynamic system and 
%  a procedure for learning to balance the pole.  

%
clear all;

SHOW_ANIMATION=0;%%if you want to visualize, set this to 1.

%%%%initialize the network
net=net_init_pole();

Parameters=[];

addpath(genpath('../CoreModules'));

SHOW_ANIMATION_Every_N=200; %every n trials
GAMMA   = 0.99;       % Discount factor for critic. 
EPSILON = 0.01; %changed from 0.05
ACTIONS = 2;
MaxUpdateDelay=50000;


MAX_FAILURES  =  5000;      % Termination criterion. 
MAX_STEPS   =     100000;

TrainErr=[];
MaxSteps=[];
steps = 0;
failures=0;
success=0;

opts.use_gpu=0;%don't use gpu for this application, since it will be slow
opts.parameters.mom =0.9;
opts.parameters.lr =1e1;
opts.parameters.weightDecay=1e-6;
opts.parameters.clip=5e-3;

% Turning on the double buffering to plot the cart and pole
if SHOW_ANIMATION
    h = figure(1);
    set(h,'doublebuffer','on')
end

% Iterate through the action-learn loop. 
while (failures < MAX_FAILURES)
    
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  RESET STARTS %%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Starting state is (0 0 0 0)
    x         = 0;       % cart position, meters 
    x_dot     = 0;       % cart velocity
    theta     = 0;       % pole angle, radians
    theta_dot = 0.0;     % pole angular velocity

    state=[x;x_dot;theta;theta_dot];
    valid=is_valid_state(x,x_dot,theta,theta_dot);
   
    InputBatch=zeros(4,MaxUpdateDelay,'like',state);
    opts.dzdy =zeros(ACTIONS,MaxUpdateDelay,'like',state);
    BatchErr=zeros(1,MaxUpdateDelay);
    
    InputBatch(:,1)=state;
    opts.samples=1;
    res(1).x=state;
    [ net,res,opts] = net_ff(net,res,opts);
    Q_new=res(end).x;
    [V_new,a_new]=max(Q_new);

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%  RESET  END %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    failed=0;
    
    while steps < MAX_STEPS && failed==0

        if SHOW_ANIMATION&&(mod(failures,SHOW_ANIMATION_Every_N)==0||success)
            plot_Cart_Pole(x,theta)
        end
        %Choose action randomly, biased by current weight. 
        r=rand(1);

        %% make a selection and report the score Q(s,a) 

        Q_old=Q_new;

        if r<EPSILON    
            a_old=randi(ACTIONS);
        else
            %select the highest scored action
            a_old=a_new;
        end


        %Apply action to the simulated cart-pole
        [x,x_dot,theta,theta_dot]=Cart_Pole(a_old-1,x,x_dot,theta,theta_dot);    
       
        state=[x;x_dot;theta;theta_dot];
        valid=is_valid_state(x,x_dot,theta,theta_dot);

        steps=steps+1;    
        InputBatch(:,steps+1)=state;
        opts.samples=steps+1;

        if valid<0	
            %Failure occurred
            failed = 1;
            failures=failures+1;
            MaxSteps=[MaxSteps;steps];


            disp(['Trial was ' int2str(failures) ' steps '  num2str(steps)]);
            

            %Reinforcement upon failure is -1. Prediction of failure is 0.
            r = -1.0;
            V_new = 0.;
            
            
            steps = 0;
            
        else
            %Not a failure.
            failed = 0;

            %Reinforcement is 0. Prediction of failure given by v weight.
            r = 0;
            
            %Value of the new state:
            %V=max_a(Q(s,a))
            %Run MLP,
            res(1).x=state;
            [net,res,opts] = net_ff(net,res,opts);
            Q_new=res(end).x;
            [V_new,a_new]=max(Q_new);


        end

        %Heuristic reinforcement is:   current reinforcement
        %     + gamma * new failure prediction - previous failure prediction
        %%derivative with L2 cost:

        der=Q_old(a_old)-(r + GAMMA * V_new);
        opts.dzdy(a_old,opts.samples-1)=der;
        BatchErr(opts.samples-1)=gather(der.^2)/2;


        if(failed==1||opts.samples==MaxUpdateDelay)
            
            %%%%%%%%%%%%%%%%%%%%%
            opts.dzdy=opts.dzdy(:,1:opts.samples-1);

            %%This is redundant, just to recalculate some intermediate values
            %%and put them into the right place.
            res(1).x=InputBatch(:,1:opts.samples-1);
            [net,res,opts] = net_ff(net,res,opts);  

            %%
            [ net,res,opts ] = net_bp( net,res,opts );    
            [ net,res,opts ] = sgd( net,res,opts );

            TrainErr=[TrainErr;mean(BatchErr(1:opts.samples-1))];
            %disp(mean(BatchErr(1:opts.samples)))

            

            opts.samples=0;


        end
        
     

    end
    if steps>=MAX_STEPS
        success=1;
        if SHOW_ANIMATION==0
            break;
        else
            steps=0;
            continue;
        end
    end

  end
  
if (failures == MAX_FAILURES)
    disp(['Pole not balanced. Stopping after ' int2str(failures) ' failures ' ]);
else
    disp(['Pole balanced successfully for at least ' int2str(steps) ' steps ' ]);
end
close all;
figure;subplot(1,2,1);plot(TrainErr);title('Training Errors');
subplot(1,2,2);plot(MaxSteps);title('Steps');