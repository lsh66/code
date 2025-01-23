dataset_name = "karate"  # name of dataset
path = "../data/datasets/" + dataset_name + ".txt"  # path to dataset


iteration = 1  # number of iterations for label selection step (mostly is set to 1 or 2)
merge_flag = 1  # merge_flag=0 -> do not merge //  merge_flag=1 -> do merge
write_flag = 0  # 1 means write nodes labels to file. 0 means do not write
modularity_flag = 1  # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1  # 1 means calculate NMI. 0 means do not calculate NMI
#%%
# ------------------------- compute nodes neighbors and nodes degree --------------------------
nodes_neighbors = {}

i = 0
with open(path) as f:
    print("hh")