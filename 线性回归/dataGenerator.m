clc
clear 
close all

% x = -10:0.01:10;
% y = -10:0.01:10;
% x_randn = randn(1, length(x))*5;
% x = x+x_randn;
% y_randn = randn(1, length(y))*5;
% y = y+y_randn;
% 
% z = 3*x+4*y+6;
% z_randn = randn(1, length(z))*10;
% z = z+z_randn;
% x = x';
% y = y';
% z = z';
% samples = [x, y, z];
% 
% save samples.mat
% scatter3(x,y,z, 'r')


% 
mat = load('./samples.mat');
mat = mat.samples;
x = mat(:, 1);
y = mat(:, 2);
z = mat(:, 3);

scatter3(x,y,z, 'r')

% z_pred = 3.03582457*x+4.0155888*y+5.54841543;
% hold on
% scatter3(x,y,z_pred, 'b')