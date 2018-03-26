
import sys
import pickle
sys.path.append("../udacity/tools/")

# py 3 file content change

# FILE word_data.pkl

originalPath = "../udacity/tools/word_data.pkl"
replacePath = "../udacity/tools/word_data.pkl"

content = ""
outsize = 0

with open(originalPath, 'rb') as infile:
    content = infile.read()
with open(replacePath, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

# FILE email_authors.pkl

originalPath = "../udacity/tools/email_authors.pkl"
replacePath = "../udacity/tools/email_authors.pkl"

content = ""
outsize = 0

with open(originalPath, 'rb') as infile:
    content = infile.read()
with open(replacePath, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Files processed")    
