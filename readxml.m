function [ mask, maxpoint, minpoint, exception, nummass ] = readxml( filename, height, width )
%READXML Summary of this function goes here
%   Detailed explanation goes here
mask = uint8(zeros(height, width));
fid = fopen(filename, 'r');
linenum = 0;
maxpoint = [];
minpoint = [];
exception = 0;
nummass = 0;
while ~feof(fid)
    line = fgetl(fid);
    linenum = linenum + 1;
    [str1, str2] = strtok(line, '<string>Mass</string>');
    if strcmp(str2, '<string>Mass</string>') == 1
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
    end
end
fclose(fid);