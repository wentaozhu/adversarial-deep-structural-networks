function [ minx, miny, maxx, maxy ] = rectbox( im )
%RECTBOX Summary of this function goes here
%   Detailed explanation goes here
minx = 1;
miny = 1;
maxx = size(im,2);
maxy = size(im,1);
for col = 1 : size(im,2)
    if sum(im(:,col)) ~= 0
        minx = col;
        for maxcol = size(im,2): -1 : col
            if sum(im(:,maxcol)) ~= 0
                maxx = maxcol;
                break;
            end
        end
        break;
    end
end
for row = 1 : size(im,1)
    if sum(im(row,:)) ~= 0
        miny = row;
        for maxrow = size(im,1) : -1 : row
            if sum(im(maxrow,:)) ~= 0
                maxy = maxrow;
                break;
            end
        end
        break;
    end
end
end

