clc
clear
close all

x = -10:0.1:10;
y1 = 3*x-10 ;
y2 = 3*x+10;
y1_n = 2*wgn(1, length(y1), 0.2);
y2_n = 2*wgn(1, length(y1), 0.2);

y1 = y1+y1_n;
y2 = y2+y2_n;
scatter(x, y1)
hold on
scatter(x, y2)
l1 = ones(1, length(y1));
l2 = ones(1, length(y2))*(-1);
y1 = [x; y1; l1]';
y2 = [x; y2; l2]';
y = [y1; y2];
save data.mat y
% 
% mat = load('./data.mat');
% mat = mat.y;
% 
% scatter(mat(1:201, 1), mat(1:201, 2), 'r')
% hold on
% scatter(mat(202:end, 1), mat(202:end, 2), 'b')
% 
% w1 = 1.22964396
% w2 = -0.42121456
% b = 0.521306112996
% 
% x = -10:0.1:10;
% 
% y = -w1/w2.*x-b/w2;
% hold on
% plot(x, y, 'g')
% 



