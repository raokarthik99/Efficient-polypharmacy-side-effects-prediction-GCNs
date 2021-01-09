from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = 'output/ppm-ggm-nn/'

feed_dict = load_data_torch("data/", et_list, mono=True)
data = Data.from_dict(feed_dict)
n_drug, n_drug_feat = data.d_feat.shape
n_prot, n_prot_feat = data.p_feat.shape
n_et_dd = len(et_list)

# GET THE ENITRE DATASET LOADED INTO THE TRAIN VARIABLES BY PASSING '1.0' AS PROBABILITY OF PICKING OBSERVATIONS FOR THE SAME
data.train_idx, data.train_et, data.train_range,data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index, 1.0)

# re-construct node feature
data.p_feat = torch.cat([dense_id(n_prot), torch.zeros(size=(n_drug, n_prot))], dim=0)
data.d_feat = dense_id(n_drug)
n_drug_feat = n_drug
n_prot_feat = n_prot

# ###################################
# dp_edge_index and range index
# ###################################
data.dp_edge_index = np.array([data.dp_adj.col-1, data.dp_adj.row-1])

count_drug = np.zeros(n_drug, dtype=np.int)
for i in data.dp_edge_index[1, :]:
    count_drug[i] += 1
range_list = []
start = 0
end = 0
for i in count_drug:
    end += i
    range_list.append((start, end))
    start = end

data.dp_edge_index = torch.from_numpy(data.dp_edge_index + np.array([[0], [n_prot]]))
data.dp_range_list = range_list

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
if iteration == 8: # REPLACE 8 WITH TOTAL NUMBER OF ITERATIONS THAT WOULD BE REQUIRED (HERE, WE ARE CONSTRAINED BY GPU MEMORY TO KEEP A VALUE LESSER THAN 8)
    final = final + 1
    number_of_side_effects = number_of_side_effects + 1

number_of_drug_combos = number_of_drugs*(number_of_drugs-1)/2

data.test_idx = torch.zeros([2, int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)
data.test_et = torch.zeros([int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)

data.test_idx[0], data.test_idx[1], data.test_et = compute_side_effect_drug_pair_combinations(number_of_drugs, number_of_side_effects, number_of_drug_combos, initial, final)
##############################################################################################################################################################################

data.d_norm = torch.ones(n_drug)
data.p_norm = torch.ones(n_prot+n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))

##### CHECK WHETHER BELOW CODE IS REALLY NECESSARY 
# data.d_feat.requires_grad = True

# source_dim = n_prot_feat
# embed_dim = 64
# target_dim = 32

# encoder = HierEncoder(source_dim, embed_dim, target_dim, n_prot, n_drug)
# # decoder = NNDecoder(target_dim, n_et_dd)
# decoder = MultiInnerProductDecoder(target_dim, n_et_dd)
# model = MyGAE(encoder, decoder)

# device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device_name = 'cpu'
# print(device_name)
# device = torch.device(device_name)

# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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

    z = model.encoder(data.p_feat, data.dp_edge_index, data.dp_range_list, data.p_norm)
    
    pos_score = model.decoder(z, data.test_idx, data.test_et)
    
    return pos_score

print('evaluating over the entire dataset and generating the results...')
with torch.no_grad():
  result = evaluate()
torch.cuda.empty_cache()

print('storing the results...')
old_data = []
if iteration > 1:
  with open('output/evaluation/ppm-ggm-nn.pkl', 'rb') as f:
    old_data = pickle.load(f)
with open('output/evaluation/ppm-ggm-nn.pkl', 'wb') as f:
    pickle.dump(old_data + result.tolist(), f)
print('Result of this iteration:', result)
print('Total number of records processed till now:', len(old_data + result.tolist()))

# with open('output/evaluation/ppm-ggm-nn.pkl', 'rb') as f:
#      a = np.array([pickle.load(f)])
# print(a)