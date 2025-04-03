import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.system("nvcc -Xptxas -O3 -gencode arch=compute_80,code=[sm_80,compute_80] main.cu -o red_patterns")