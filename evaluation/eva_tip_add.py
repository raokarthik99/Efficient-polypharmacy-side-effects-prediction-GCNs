from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = 'output/tip-add/'

feed_dict = load_data_torch("data/", et_list, mono=True)

data = Data.from_dict(feed_dict)
data.n_drug = data.d_feat.shape[0]
data.n_prot = data.p_feat.shape[0]
data.n_dd_et = len(et_list)

data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.dd_test_idx, data.dd_test_et, data.dd_test_range = process_edges(data.dd_edge_index, 1.0)
data.pp_train_indices, data.pp_test_indices = process_prot_edge(data.pp_adj, 1.0)

data.d_feat = sparse_id(data.n_drug)
data.p_feat = sparse_id(data.n_prot)
data.n_drug_feat = data.d_feat.shape[1]
data.d_norm = torch.ones(data.n_drug_feat)

# ###################################
# dp_edge_index and range index
# ###################################
data.dp_edge_index = np.array([data.dp_adj.col-1, data.dp_adj.row-1])

count_drug = np.zeros(data.n_drug, dtype=np.int)
for i in data.dp_edge_index[1, :]:
    count_drug[i] += 1
range_list = []
start = 0
end = 0
for i in count_drug:
    end += i
    range_list.append((start, end))
    start = end

data.dp_edge_index = torch.from_numpy(data.dp_edge_index + np.array([[0], [data.n_prot]]))
data.dp_range_list = torch.Tensor(range_list)

###########################################################################################################################################################################
# COMPUTE ALL POSSIBLE SIDE-EFFECT:DRUG PAIR COMBINATIONS
############################################################################################################################################################################
# FOR A SPECIFIC DRUG PAIR (A,B), 
# HARDCODE A AND B AS VALUES FOR data,test_idx[0][k] and data.test_idx[1][k] respectively 

# FOR A SPECIFIC SIDE EFFECT (S),
# HARDCODE S AS VALUE FOR data.test_et[0]

iteration = int(input('Enter iteration counter to compute next subset of results: '))

# total_number_of_drugs = 645
number_of_drugs = 645

# number_of_side_effects =  int(input('Enter number of side-effects to process for this iteration '))
number_of_side_effects = 137
# total_number_of_side_effects = 1097

initial = (iteration - 1) * number_of_side_effects
final = iteration * number_of_side_effects
if iteration == 8: # REPLACE 8 WITH TOTAL NUMBER OF ITERATIONS THAT WOULD BE REQUIRED (HERE, WE ARE CONSTRAINED BY GPU MEMORY TO KEEP A VALUE LESSER THAN 4)
    final = final + 1
    number_of_side_effects = number_of_side_effects + 1

number_of_drug_combos = number_of_drugs*(number_of_drugs-1)/2

data.dd_test_idx = torch.zeros([2, int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)
data.dd_test_et = torch.zeros([int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)

data.dd_test_idx[0], data.dd_test_idx[1], data.dd_test_et = compute_side_effect_drug_pair_combinations(number_of_drugs, number_of_side_effects, number_of_drug_combos, initial, final)
##############################################################################################################################################################################


##### CHECK WHETHER BELOW CODE IS REALLY NECESSARY 
# device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device_name)
# device = torch.device(device_name)

# encoder = FMEncoder(device, data.n_drug_feat, data.n_dd_et, data.n_prot,
#                         data.n_prot, data.n_drug)
# decoder = MultiInnerProductDecoder(encoder.out_dim, data.n_dd_et)
# model = MyGAE(encoder, decoder)

# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# data = data.to(device)


#########################################################################
# Load Trained Model
#########################################################################
print('loading the trained model...')
filepath_model = os.path.join(out_dir, '100ep_model.pb')
model = torch.load(filepath_model)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)

#########################################################################
# Evaluation and Record
#########################################################################
def evaluate():
    model.eval()
    
    # NOTE THAT HERE TRAIN VARIABLES ARE ACTUALLY REPRESENTATIVE OF THE ENTIRE DATASET
    # NECESSARY FOR PREDICTING/EVALUATING ALL POSSIBLE SIDE-EFFECT/DRUG-PAIR COMBINATIONS  
    # THIS IS NOT A RANDOM SUBSET LIKE THAT WE GENERATED DURING MODEL TRAINING  

    z = model.encoder(data.d_feat, data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.d_norm, data.p_feat, data.pp_train_indices, data.dp_edge_index, data.dp_range_list)
    
    pos_score = model.decoder(z, data.dd_test_idx, data.dd_test_et)
    
    return pos_score

print('evaluating over the entire dataset and generating the results...')
with torch.no_grad():
  result = evaluate()
torch.cuda.empty_cache()

print('storing the results...')
old_data = []
if iteration > 1:
  with open('output/evaluation/tip-add.pkl', 'rb') as f:
    old_data = pickle.load(f)
with open('output/evaluation/tip-add.pkl', 'wb') as f:
    pickle.dump(old_data + result.tolist(), f)
print('Result of this iteration:', result)
print('Total number of records processed till now:', len(old_data + result.tolist()))

# import numpy as np 
# import pickle
# with open('output/evaluation/tip-add.pkl', 'rb') as f:
#      a = np.array([pickle.load(f)])
# print(a)