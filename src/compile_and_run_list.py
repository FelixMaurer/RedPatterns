import os
import sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= str(sys.argv[1])
#os.system("module load compiler/GCC")
#os.system("module load system/CUDA")
#os.system("nvcc -Xptxas -O3 -arch=sm_75 -o red_patterns main.cu")
paramFileName = str(sys.argv[1])
os.system("nvcc -Xptxas -O3 -gencode arch=compute_80,code=[sm_80,compute_80] main.cu -o red_patterns")
os.system("chmod 777 red_patterns")
os.system("chmod 777 " + paramFileName)

file = open(paramFileName, 'r')
Lines = file.readlines()
paramDir = paramFileName[0:len(paramFileName)-4]
parentDir = "/lustre/project/nhr-rbc-pattern"
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    outDir = parentDir + "/" + paramDir + "/" + "parameters_" + str(count).zfill(3)
    os.system("mkdir " + parentDir + "/" + paramDir)
    os.system("mkdir " + outDir)
    os.system("cp " + parentDir + "/" + "red_patterns " + outDir)
    os.system("cd " + outDir)
    #print("running parameters {}: {}".format(count, line.strip()))
    commandStr = outDir + "/red_patterns " + line.strip()
    #print("passing command: " + commandStr)
    os.system(commandStr)
    #os.system("mv  -v ./*.dat" + " ./" + outDir)
    os.system("cd " + parentDir)