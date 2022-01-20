folder = "output/3_bunny_high_res/trajectory_soft_GN_load_/epoch_0/";

file = fopen(folder + "A.txt", 'r');
triplets = fscanf(file, '%d %d %le', [3 Inf]);
fclose(file);

A = sparse(triplets(1, :), triplets(2, :), triplets(3, :));

file = fopen(folder + "residual.txt", 'r');
b = fscanf(file, "%le %le %le", [3 inf]);
b  = reshape(b, [], 1);

n_frame = 100;
n_vert = size(b, 1) / n_frame / 3;

M = A(1:n_vert * 3, (n_frame - 2) * n_vert * 3 + 1: (n_frame - 1) * n_vert * 3);
A0 = A(1: n_vert * 3, 1: n_vert * 3);

ATA = transpose(A) * A;
projected_ATA = ATA(1:(n_frame - 2) * n_vert * 3, 1:(n_frame - 2) * n_vert * 3);
ATb = transpose(A) * b;
projected_ATb = ATb(1:(n_frame - 2) * n_vert * 3);