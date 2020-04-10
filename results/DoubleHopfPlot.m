clear;
%% Region 1 2 3
% load('Region1_7p0_7p8_EfdNoLim_200sec.mat'); R1 = data;
% load('Region2_6p7_7p7_EfdNoLim_320sec.mat'); R2 = data;
% load('Region3_6p7_7p2_EfdNoLim_200sec_GotoZero.mat'); R3s = data; 
% load('Region3_6p7_7p2_EfdNoLim_35sec_GoUnstable.mat'); R3u = data;
% clear data
% subplot(2,2,1);plot(R1(:,1),R1(:,[5,12]));xlim([0,183]);ylim([-0.008,0.015]);xlabel('t(sec)');title('Region 1');legend('G1 Speed','G2 Speed');
% subplot(2,2,2);plot(R2(:,1),R2(:,[5,12]));xlim([0,325]);ylim([-0.008,0.015]);xlabel('t(sec)');title('Region 2');
% subplot(2,2,3);plot(R3s(:,1),R3s(:,[5,12]));xlim([0,200]);ylim([-0.001,0.001]);xlabel('t(sec)');title('Region 3(a)');
% subplot(2,2,4);plot(R3u(:,1),R3u(:,[5,12]));xlim([0,35]);ylim([-0.01,0.01]);xlabel('t(sec)');title('Region 3(b)');

% plot(R1(:,1),R1(:,[5,12]));xlim([0,183]);ylim([-0.008,0.015]);xlabel('t(sec)');ylabel('p.u.');title('Region 1');legend('G1 Speed','G2 Speed');
% plot(R2(:,1),R2(:,[5,12]));xlim([0,325]);ylim([-0.008,0.015]);xlabel('t(sec)');ylabel('p.u.');title('Region 2');legend('G1 Speed','G2 Speed');
% subplot(2,1,1);plot(R3s(:,1),R3s(:,[5,12]));xlim([0,200]);ylim([-0.001,0.001]);xlabel('t(sec)');ylabel('p.u.');title('Region 3(a)');legend('G1 Speed','G2 Speed');
% subplot(2,1,2);plot(R3u(:,1),R3u(:,[5,12]));xlim([0,35]);ylim([-0.01,0.01]);xlabel('t(sec)');ylabel('p.u.');title('Region 3(b)');
%% Region 4
% load('Region4_7p1_7p3_EfdNoLim_1000sec_StableLC.mat'); R4 = data; clear data
% plot(R4(:,1),R4(:,[5,12]));ylim([-0.005,0.005]);xlabel('t(sec)');title('Region 4');legend('G1 Speed','G2 Speed');
% axes('position',[.2 .65 .2 .2]);%set(gca,'fontsize',30);legend('LS','TSAT',10);
% plot(R4(:,1),R4(:,[5,12]));axis([948,952,-0.005,0.005]);
% 
% plot(R4(end-1000:end,5),R4(end-1000:end,12));title('Phase portrait of \omega_1 - \omega_2');xlabel('\omega_1(p.u.)');ylabel('\omega_2(p.u.)');
%% Region 5
% load('Region5_7p1_7p4_4ksec.mat'); R5 = data; clear data
% 
% plot(R5(:,1),R5(:,5));axis([0,2500,-0.005,0.010]);xlabel('t(sec)');title('Region 5');legend('G1 Speed');
% axes('position',[.2 .65 .2 .2]);%set(gca,'fontsize',30);legend('LS','TSAT',10);
% plot(R5(:,1),R5(:,5));axis([600,605,-0.005,0.005]);
% axes('position',[.65 .2 .2 .2]);%set(gca,'fontsize',30);legend('LS','TSAT',10);
% plot(R5(:,1),R5(:,5));axis([2250,2260,-0.005,0.005]);
% 
% tstart = 400000;
% plot3(R5(tstart:end,5),R5(tstart:end,12),R5(tstart:end,4));
% xlabel('G1 Speed')
% ylabel('G2 Speed')
% zlabel('G1 Angle')

%% Region 6
% load('Region6_7p2_7p45_EfdNoLim_Omg+-0p006_70sec.mat'); R6 = data; clear data
% plot(R6(:,1),R6(:,[5,12]));axis([0,67,-0.02,0.05]);xlabel('t(sec)');title('Region 6');legend('G1 Speed','G2 Speed');

%% Region 7
load('Region7_7p3_7p6_EfdNoLim_160sec.mat'); R7 = data; clear data
plot(R7(:,1),R7(:,[5,12]));axis([0,154,-0.02,0.05]);xlabel('t(sec)');ylabel('p.u.');title('Region 7');legend('G1 Speed','G2 Speed');

%% Region 6'
load('7p0_7p5007_10ksec.mat'); R6p = data; clear data
plot(R6p(:,1),R6p(:,5));axis([0,6500,-0.004,0.004]);xlabel('t(sec)');title('Region 6\prime');legend('G1 Speed');
axes('position',[.2 .65 .2 .2]);%set(gca,'fontsize',30);legend('LS','TSAT',10);
plot(R6p(:,1),R6p(:,5));axis([700,705,-0.004,0.004]);
axes('position',[.65 .2 .2 .2]);%set(gca,'fontsize',30);legend('LS','TSAT',10);
plot(R6p(:,1),R6p(:,5));axis([1900,1910,-0.002,0.002]);