function ret = SphereGenerate(center, amount, label)
x0 = center(1);
y0 = center(2);
z0 = center(3);

w = rand(amount, 3);
x = w(:, 1)+x0;
y = w(:, 2)+y0;
z = w(:, 3)+z0;
labels = ones(amount, 1)*label;
ret = [x, y, z, labels];
