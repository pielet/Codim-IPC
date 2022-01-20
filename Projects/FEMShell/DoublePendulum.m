function DoublePendulumGUI
    figure1=figure(1);
    set(figure1,'position',get(0,'screensize'),...
        'unit','normalized','color',[0,0,0])
    axes1=axes;
    set(axes1,'unit','normalized',...
        'position',[0.1,0.1,0.6,0.8],...
        'color',[0,0,0])
    singleB=uicontrol('Style','pushbutton',...
        'unit','normalized',...
        'position',[0.8,0.8,0.05,0.1],...
        'string','single','fontsize',20,...
        'Callback',@singleBfcn,...
        'BackgroundColor',[0,0,0],'foregroundcolor',[1,1,1]);
    multiB=uicontrol('Style','pushbutton',...
        'unit','normalized',...
        'position',[0.9,0.8,0.05,0.1],...
        'string','multi','fontsize',20,...
        'Callback',@multiBfcn,...
        'BackgroundColor',[0,0,0],'foregroundcolor',[1,1,1]);
    angleT=uicontrol('Style','text',...
        'unit','normalized',...
        'position',[0.8,0.7,0.15,0.05],...
        'string','ANGLE','fontsize',20,...
        'BackgroundColor',[0,0,0],'foregroundcolor',[1,1,1]);
    angle1S=uicontrol('Style','slider',...
        'unit','normalized',...
        'position',[0.75,0.6,0.25,0.05],...
        'Callback',@angle1Sfcn,...
        'Min',-pi,'Max',pi);
    angle2S=uicontrol('Style','slider',...
        'unit','normalized',...
        'position',[0.75,0.55,0.25,0.05],...
        'Callback',@angle2Sfcn,...
        'Min',-pi,'Max',pi);
    playB=uicontrol('Style','pushbutton',...
        'unit','normalized',...
        'position',[0.8,0.4,0.15,0.1],...
        'string','Start/Pause','fontsize',20,...
        'Callback',@playBfcn);
    CloseB=uicontrol('Style','pushbutton',...
        'unit','normalized',...
        'position',[0.8,0.2,0.15,0.1],...
        'string','Close','fontsize',20,...
        'Callback',@CloseBfcn);
    ​
    % Get the size of the axes
    set(axes1,'unit','pixel');
    AxPos=get(axes1,'position');
    woverh=AxPos(3)/AxPos(4);
    set(axes1,'unit','normalized');
    ​
    % Initialize
    th01=0.1;
    th02=0.2;
    Play=0;
    Closed=0;
    Single=1;
    theta1=th01;
    theta2=th02;
    MultiNum=50;
    ​
    InitialData=[theta1;theta2;zeros(size(theta1));zeros(size(theta1))];
    ​
    dt=0.01;
    g=50;
    [M,N]=size(InitialData);
    cmap1=colormap('jet');
    cmap=spline(1:64,cmap1',linspace(1,64,MultiNum))';
    cmap(cmap<0)=0;
    cmap(cmap>1)=1;
    ​
    ​
    u0=InitialData;
    u1=u0;Plot
    %Solve ODE
    while Closed==0
        pause(0.05)
        while Play==1 && Closed==0
            k1=odefunction(u0);
            k2=odefunction(u0+1/2*dt*k1);
            k3=odefunction(u0+1/2*dt*k2);
            k4=odefunction(u0+dt*k3);
            u1=u0+1/6*dt*(k1+2*k2+2*k3+k4);
            %set(gca,'ColorOrder',cmap,'NextPlot','replacechildren');
            Plot
            u0=u1;
            th1=mod(u0(1,1)+pi,2*pi)-pi;
            th2=mod(u0(2,1)+pi,2*pi)-pi;
            set(angle1S,'Value',th1);
            set(angle2S,'Value',th2);
        end
    end
    close(figure1)
    ​
    % ODE Function
    function uut=odefunction(uu)
        uut=uu;
        T1=uu(1,:);
        T2=uu(2,:);
        T1t=uu(3,:);
        T2t=uu(4,:);
        Delta=4/9-1/4*(cos(T2-T1)).^2;
        uut(1,:)=T1t;
        uut(2,:)=T2t;
        uut(3,:)=1./Delta.*(1/6*T2t.^2.*sin(T2-T1)-1/2*g.*sin(T1)...
            +1/4*T1t.^2.*sin(T2-T1).*cos(T2-T1)+1/4*g.*sin(T2).*cos(T2-T1));
        uut(4,:)=1./Delta.*(-1/4*T2t.^2.*sin(T2-T1).*cos(T2-T1)+3/4*g.*sin(T1).*cos(T2-T1)...
            -2/3*T1t.^2.*sin(T2-T1)-2/3*g.*sin(T2));
    end
    function singleBfcn(hObj,eventdata)
        Single=1;
        Play=0;
        theta1=th01;
        theta2=th02;
        InitialData=[theta1;theta2;zeros(size(theta1));zeros(size(theta1))];
            u0=InitialData;
            u1=u0;
            Plot
            th1=mod(u0(1,1)+pi,2*pi)-pi;
            th2=mod(u0(2,1)+pi,2*pi)-pi;
    set(angle1S,'Value',th1);
    set(angle2S,'Value',th2);
    end
    function multiBfcn(hObj,eventdata)
        Single=0;
        Play=0;
        th1=th01;
        th2=th02;
        theta1=linspace(th1,th1+pi/180,MultiNum);
        theta2=linspace(th2,th2+pi/180,MultiNum);
        InitialData=[theta1;theta2;zeros(size(theta1));zeros(size(theta1))];
            u0=InitialData;
            u1=u0;
            Plot
            th1=mod(u0(1,1)+pi,2*pi)-pi;
            th2=mod(u0(2,1)+pi,2*pi)-pi;
    set(angle1S,'Value',th1);
    set(angle2S,'Value',th2);
    end
    function angle1Sfcn(hObj,eventdata)
        %if Play==0
            if Single==1
                theta1=get(hObj,'Value');
            else
                th1=get(hObj,'Value');
                theta1=linspace(th1,th1+pi/180,MultiNum);
            end
            InitialData=[theta1;theta2;zeros(size(theta1));zeros(size(theta1))];
            u0=InitialData;
            u1=u0;
            Plot
        %end
    end
    function angle2Sfcn(hObj,eventdata)
        %if Play==0
            if Single==1
                theta2=get(hObj,'Value');
            else
                th2=get(hObj,'Value');
                theta2=linspace(th2,th2+pi/180,MultiNum);
            end
            InitialData=[theta1;theta2;zeros(size(theta1));zeros(size(theta1))];
            u0=InitialData;
            u1=u0;
            Plot
        %end
    end
    function CloseBfcn(hObj,eventdata)
        Closed = 1;
    end
    function playBfcn(hObj,eventdata)
        if Play==0
            Play=1;
        else
            Play=0;
        end
    end
    function Plot
        [M,N]=size(u1);
        set(gca,'ColorOrder',cmap,'NextPlot','replacechildren');
    plot([zeros(N,1),sin((u1(1,:))'),sin((u1(1,:))')+sin((u1(2,:))')]',...
        [zeros(N,1),-cos((u1(1,:))'),-cos((u1(1,:))')-cos((u1(2,:))')]',...
        'LineWidth',25)
    
    axis([-3*woverh,3*woverh,-3,3])
    drawnow;
    end
end