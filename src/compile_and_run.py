import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.system("nvcc -Xptxas -O3 -arch=sm_75 -o red_patterns main.cu")
os.system("./red_patterns")