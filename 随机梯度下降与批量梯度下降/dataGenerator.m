clc
clear all
close all

x = -20:0.01:20;
line = 2.5*x+3.5;

x_rand = randn(1, length(line))*10;
line = line+x_rand;

x = x';
line = line';

samples = [x, line];

save samples
