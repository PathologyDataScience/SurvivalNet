function A = text2cell(filename, delimiter)

fid = fopen(filename, 'r');

if(fid >= 3) %file opened successfully
    %get first line, determine number of columns
    line = fgetl(fid);
    header = ScanFirst(line, delimiter);
    columns = length(header);
      
    %create format string
    format = [];
    for i = 1:columns-1
        format = [format '%s '];
    end
    format = [format '%s'];
    
    %scan remainder of file
    text = textscan(fid, format, 'delimiter', delimiter);
    
    %allocate output
    A = cell(length(text{1})+1,columns);
    
    %put header in first line of 'A'
    A(1,:) = {header{:}};
    
    %put data into remaining lines of 'A'
    A(2:end, :) = [text{:}];
    
    %close file
    fclose(fid);
    
else %file error
    A = {};
end

function Tokens = ScanFirst(String, delimiter)
%used to read in first line of file, since Matlab's textscan will discard
%the last 
%inputs:
%String - string to be separated into tokens.
%delimiter - character delimiter to separate 'String' into tokens.
%outputs:
%Tokens - cell array of strings, extracted from 'String'.

%get indexes of delimiters
Delimiters = strfind(String, sprintf(delimiter));

%initialize output
Tokens = cell(1,length(Delimiters)+1);

%check number of entries
if(length(Tokens) > 1)
    
    %loop through, collecting tokens
    for i = 1:length(Tokens)-1
        
        %read strings between delimiters
        if(i == 1)
            Tokens{1} = String(1:Delimiters(i)-1);
        else
            Tokens{i} = String(Delimiters(i-1)+1:Delimiters(i)-1);
        end
        
        %trim
        Tokens{1} = strtrim(Tokens{1});
    
    end
    
    %capture last token
    Tokens{end} = String(Delimiters(end)+1:end);
    
    %trim
    Tokens{end} = strtrim(Tokens{end});
    
else
    
    %remove leading, trailing whitespace and return
    Tokens{1} = strtrim(String);
    
end