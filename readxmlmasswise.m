function [ maskarr, maxpoint, minpoint, exception, nummass ] = readxmlmasswise( filename, height, width, flag )
%READXML Summary of this function goes here
%   Detailed explanation goes here
fid = fopen(filename, 'r');
linenum = 0;
maxpoint = [];
minpoint = [];
exception = 0;
nummass = 0;
maskarr = {};
while ~feof(fid)
    line = fgetl(fid);
    linenum = linenum + 1;
    [str1, str2] = strtok(line, '<string>Mass</string>');
    if strcmp(str2, '<string>Mass</string>') == 1
        mask = uint8(zeros(height, width));
        prex = -1; prey = -1; headx = -1; heady = -1;
        nummass = nummass + 1;
        while ~feof(fid)
            line = fgetl(fid);
            [str1, str2] = strtok(line, '<key>Point_mm</key>');
            if strcmp(str2, '<key>Point_mm</key>') == 1
                break;
            end
        end
        while ~feof(fid)
            line = fgetl(fid);
            [str1, str2] = strtok(line, '<key>Point_px</key>');
            if strcmp(str2,'<key>Point_px</key>') == 1
                line = fgetl(fid); % <array>
                linenum = linenum + 1;
                line = fgetl(fid);
                linenum = linenum + 1;
                [str1, str2] = strtok(line, '</array>');
                minx = width;
                miny = height; 
                maxx = 0; 
                maxy = 0;
                while strcmp(str2, '</array>') ~= 1
                    [str1, str2] = strtok(line, '<string>');
                    [str1, str2] = strtok(str2, '<string>');
                    [str1, str2] = strtok(str1, ', ');
                    xx = str2double(str1(2:end));
                    yy = str2double(str2(3:end-1));
                    xx = int32(xx);
                    yy = int32(yy);
                    if xx == 0
                        xx = xx + 1;
                    end
                    if yy == 0
                        yy = yy + 1;
                    end
                    if xx > width
                        xx = width;
                    end
                    if  yy > height
                        yy = height;
                    end
                    if xx < 0 || yy < 0 || xx > width || yy > height
                        %xx
                        %yy
                        exception  = 1;
                        %linenum
                        %str1
                        %str2
                        continue;
                    end 
                    %if strcmp(filename, '.\AllXML\22670094.xml') == 1
                    %    tmp = xx;
                    %    xx = yy;
                    %    yy = tmp;
                    %end
                    if minx > xx
                        minx = xx;
                    end
                    if maxx < xx
                        maxx = xx;
                    end
                    if miny > yy
                        miny = yy;
                    end
                    if maxy < yy
                        maxy = yy;
                    end
                    mask(yy, xx) = 255;
                    if prex ~= -1 && prey ~= -1 && strcmp(flag, 'linear')==1
                        if abs(yy-prey) <= abs(xx-prex)
                            for xindex = prex : sign(xx-prex) : xx
                                yindex = double(prey) + (double(yy-prey)*1.0/double(xx-prex))*double(xindex-prex);
                                if int32(yindex) == 0
                                    yindex = 1;
                                end
                                mask(int32(yindex), xindex) = 255;
                            end
                        elseif strcmp(flag, 'linear') == 1
                            for yindex = prey : sign(yy-prey) : yy
                                xindex = double(prex) + (double(xx-prex)*1.0/double(yy-prey))*double(yindex-prey);
                                if int32(xindex) == 0
                                    xindex = 1;
                                end
                                mask(yindex, int32(xindex)) = 255;
                            end
                        end
                    else
                        headx = xx; heady = yy;
                    end
                    prex = xx; prey = yy;
                    line = fgetl(fid);
                    linenum = linenum + 1;
                    [str1, str2] = strtok(line, '</array/');
                end
                %if maxy - miny < 16 || maxx - minx < 16
                %    continue;
                %end
                maxpoint = [maxpoint ; maxx, maxy];
                minpoint = [minpoint ; minx, miny];
                break;
            end
        end
        if abs(heady-prey) <= abs(headx-prex) && strcmp(flag, 'linear')==1
            for xindex = prex : sign(headx-prex) : headx
                yindex = double(prey) + (double(heady-prey)*1.0/double(headx-prex))*double(xindex-prex);
                if int32(yindex) == 0
                    yindex = 1;
                end
                mask(int32(yindex), xindex) = 255;
            end
        elseif strcmp(flag, 'linear') == 1
            for yindex = prey : sign(heady-prey) : heady
                xindex = double(prex) + (double(headx-prex)*1.0/double(heady-prey))*double(yindex-prey);
                if int32(xindex) == 0
                    xindex = 1;
                end
                mask(yindex, int32(xindex)) = 255;
            end
        end
        maskarr = [maskarr; mask];
    end
end
fclose(fid);