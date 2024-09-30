from datasets import load_dataset

# Carrega el dataset
ds = load_dataset("williamkgao/bookcorpus100mb")


for i in range(10000):
    print(ds['train'][i])
