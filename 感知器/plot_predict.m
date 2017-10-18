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
% 
% x = -1:0.1:3;
% y = -2:0.1:3;
% 
% w1 = 1.8744104;
% w2 = -0.09249913;
% w3 = -0.01825955;
% b = 0.113957917693;
% [x, y] = meshgrid(x, y);
% z_pred = -w1/w3*x-w2/w3*y-b/w3;
% 
% hold on
% mesh(x, y, z_pred);





