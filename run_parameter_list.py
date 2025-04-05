import os
import sys

paramFileName = str(sys.argv[1])
os.system("chmod 777 red_patterns")
os.system("chmod 777 " + paramFileName)

file = open(paramFileName, 'r')
Lines = file.readlines()
paramDir = paramFileName[0:len(paramFileName)-4]
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    outDir = "parameters_" + str(count).zfill(3)
    os.system("mkdir " + outDir)
    print("running parameters {}: {}".format(count, line.strip()))
    commandStr = "./red_patterns " + line.strip()
    print("passing command: " + commandStr)
    os.system(commandStr)
    os.system("mv  -v ./*.dat" + " ./" + outDir)