function [ outim ] = ballimenhance( I )
%BALLIMENHANCE Summary of this function goes here
%   Detailed explanation goes here
% blocksize = 16; % 2 4 8 16
% figure, imshow(I);
Icl = adapthisteq(I);
% figure, imshow(Icl);
I2 = NI(I);
% figure, imshow(I2);
mu = mean(I2(:));
I3 = NI(Icl) .* (1 - exp(-I2/mu));
% figure, imshow(I3);
I3 = NI(I3);
% figure, imshow(I3);
% M2 = RM(I3, blocksize, blocksize);
% figure, imshow(M2);
I4 = histeq(NI(I3));
% figure, imshow(I4);
I5 = I3; %NI(I3 + RM(M2, blocksize, blocksize));
% figure, imshow(I5);
I6 = histeq(NI(I5)) .* I4;
% figure, imshow(I6);
E = NI(I6);
outim = E;
% figure, imshow(E);
% P = ImToPolar(E, 0, 1, size(I,1), size(I,2));%imgpolarcoord(E,40,40); %polarim(E);
% figure, imshow(P);
% mur = mean(P, 1);
% mumax = max(mur);
% sigma = 0;
% for r = 1 : length(mur)
%     if mur(r) >= 0.4 * mumax
%         sigma = r;
%         break;
%     end
% end
% G = zeros(size(P,1), size(P,2));
% for r = 1 : size(P,1)
%     for theta = 1 : size(P,2)
%         if r <= sigma
%             G(r, theta) = 1;
%         else
%             G(r, theta) = exp(-0.5*(((r-sigma)/double(sigma))^2));
%         end
%     end
% end
% figure, imshow(G);
% P = P .* G;
% figure, imshow(P);
% P = (P-mean(P(:))) / std(P(:));
% figure, imshow(P);
% outim = PolarToIm(P, 0, 1, size(I,1), size(I,2));
% figure, imshow(outim);
end

function [N] = NI(X)
N = X - min(X(:));
N = double(N) / double(max(N(:)));
end
function [N] = RM(X, b1, b2)
% b1 for height, b2 for width
M = zeros(size(X,1)-b1+1, size(X,2)-b2+1);
for i = 1 : size(X,1)-b1+1
    for j = 1 : size(X,2)-b2+1
        block = X(i:i+b1-1, j:j+b2-1);
        M(i,j) = mean(block(:));
    end
end
N = imresize(M, [size(X,1), size(X,2)]);
end
function [impolar] = polarim(im)
X0=size(im,1)/2; Y0=size(im,2)/2;
[Y, X, z]=find(im);
X=X-X0; Y=Y-Y0;
theta = atan2(Y,X);
rho = sqrt(X.^2+Y.^2);
% Determine the minimum and the maximum x and y values:
rmin = min(rho); tmin = min(theta);
rmax = max(rho); tmax = max(theta);
% Define the resolution of the grid:
rres=128; % # of grid points for R coordinate. (change to needed binning)
tres=128; % # of grid points for theta coordinate (change to needed binning)
F = TriScatteredInterp(rho,theta,z,'natural');
%Evaluate the interpolant at the locations (rhoi, thetai).
%The corresponding value at these locations is Zinterp:
[rhoi,thetai] = meshgrid(linspace(rmin,rmax,rres),linspace(tmin,tmax,tres));
impolar = F(rhoi,thetai);
end