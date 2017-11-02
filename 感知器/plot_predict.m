clc
clear
close all

path = './samples.mat';
mat = load(path);
mat = mat.samples;

pos = mat(1:100, 1:3);
neg = mat(101:200, 1:3);

scatter3(pos(:, 1), pos(:, 2),pos(:, 3), 'r')
hold on
scatter3(neg(:, 1), neg(:, 2),neg(:, 3), 'b')


x = -1:0.1:3;
y = -2:0.1:3;

w1 = 0.53458239
w2 = 0.89768524
w3 =  0.50972598
b = 0.445783200096
[x, y] = meshgrid(x, y);
z_pred = -w1/w3*x-w2/w3*y-b/w3;

hold on
mesh(x, y, z_pred);

% close all
% clear 
% clc
% data = 'G:\电子工程\人工智能\研究工作\Movine-View Skeletons Recognition\refined_model\exp_test\reg.mat';
% mat = load(data);
% front_d = mat.front_d;
% front_g = mat.front_g;
% side_d = mat.side_d;
% fine = mat.fine;
% plot(front_d, 'r')
% hold on
% plot(front_g, 'g')
% hold on
% plot(side_d, 'b')
% hold on
% plot(fine, 'm')
% 
% % plot(mat)
% grid on
% set(gca,'xtick',[0:5:180],'xticklabel',[-90:5:90])   % 修改刻度标签
% xlabel('degree')
% legend('front_d','front_g', 'side_d', 'fine');


% close all
% clear 
% clc
% data = 'G:\电子工程\人工智能\研究工作\Movine-View Skeletons Recognition\refined_model\exp_test\data.mat';
% mat = load(data);
% fine = mat.loss_fine;
% loss = mat.loss;
% 
% plot(fine, 'r')
% hold on 
% plot(loss, 'b')
% grid on
% set(gca,'xtick',[0:5:180],'xticklabel',[-90:5:90])   % 修改刻度标签

