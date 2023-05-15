import os

# define the directory path where the text files are stored
dir_path = 'numbers/labels'

# define the value to check against


# loop through each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):  # only consider text files
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r') as f:
            # read the contents of the file into a list of lines
            lines = f.readlines()
        
        # loop through each line and check the first word
        for i, line in enumerate(lines):
            words = line.strip().split()  # split the line into words
            
            if words[0] == '16':
                words[0] = '15'
                lines[i] = ' '.join(words) + '\n'
            elif words[0] == '17':
                words[0] = '16'
                lines[i] = ' '.join(words) + '\n'
            elif words[0] == '20':
                words[0] = '17'
                lines[i] = ' '.join(words) + '\n'
            elif words[0] == '21':
                words[0] = '18'
                lines[i] = ' '.join(words) + '\n'
            elif words[0] == '25':
                words[0] = '19'
                lines[i] = ' '.join(words) + '\n'
        
       
        with open(filepath, 'w') as f:
            f.writelines(lines)
