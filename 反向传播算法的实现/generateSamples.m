clc
clear all
close all
num = 100;
c1 = SphereGenerate([1,1,1], num, 1);
c2 = SphereGenerate([-1,2,-7], num, 0);
samples = [c1;c2];
save samples

scatter3(c1(:,1),c1(:,2),c1(:,3),'r')
hold on
scatter3(c2(:,1),c2(:,2),c2(:,3),'b')


