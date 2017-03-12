clear all; close all; clc;
fid = fopen('inbreastcrfcomb.txt', 'r');
lineindex = 1;
testdi = [];
while ~feof(fid)
    line = fgetl(fid);
    if mod(lineindex, 2) == 1
        lineindex = lineindex + 1;
        continue;
    end
    lineindex = lineindex + 1;
    [str1, str2] = strtok(line);
    [str3, str4] = strtok(str2);
    [str4, str5] = strtok(str4);
    testdi = [testdi ; str2num(str5)];
end
plot(testdi);
max(testdi)