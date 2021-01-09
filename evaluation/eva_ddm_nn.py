from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = 'output/ddm-nn/'

feed_dict = load_data_torch("data/", et_list, mono=True)

[n_drug, n_feat_d] = feed_dict['d_feat'].shape
n_et_dd = len(et_list)

data = Data.from_dict(feed_dict)

# GET THE ENITRE DATASET LOADED INTO THE TRAIN VARIABLES BY PASSING '1.0' AS PROBABILITY OF PICKING OBSERVATIONS FOR THE SAME
data.train_idx, data.train_et, data.train_range, data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index, 1.0)

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
number_of_side_effects = 274
# total_number_of_side_effects = 1097

initial = (iteration - 1) * number_of_side_effects
final = iteration * number_of_side_effects
if iteration == 4: # REPLACE 4 WITH TOTAL NUMBER OF ITERATIONS THAT WOULD BE REQUIRED (HERE, WE ARE CONSTRAINED BY GPU MEMORY TO KEEP A VALUE LESSER THAN 4)
    final = final + 1
    number_of_side_effects = number_of_side_effects + 1

number_of_drug_combos = number_of_drugs*(number_of_drugs-1)/2

data.test_idx = torch.zeros([2, int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)
data.test_et = torch.zeros([int(number_of_side_effects * number_of_drug_combos)], dtype=torch.int64)

data.test_idx[0], data.test_idx[1], data.test_et = compute_side_effect_drug_pair_combinations(number_of_drugs, number_of_side_effects, number_of_drug_combos, initial, final)
##############################################################################################################################################################################

data.d_feat = dense_id(n_drug)
n_feat_d = n_drug
data.x_norm = torch.ones(n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))

# INCLUDING ENCODER, DECODER CLASS DEFINITIONS TO SATISY NAMESPACE CONSTRAINTS
class Encoder(torch.nn.Module):

    def __init__(self, in_dim, num_et, num_base):
        super(Encoder, self).__init__()
        self.num_et = num_et

        self.embed = Param(torch.Tensor(in_dim, n_embed))
        self.rgcn1 = MyRGCNConv2(n_embed, n_hid1, num_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_et, num_base, after_relu=True)

        self.reset_paramters()

    def forward(self, x, edge_index, edge_type, range_list, x_norm):
        x = torch.matmul(x, self.embed)
        x = x / x_norm.view(-1, 1)
        # x = self.rgcn1(x, edge_index, edge_type, range_list)
        x = checkpoint(self.rgcn1, x, edge_index, edge_type, range_list)
        x = F.relu(x, inplace=True)
        x = self.rgcn2(x, edge_index, edge_type, range_list)
        # x = F.relu(x, inplace=True)
        return x

    def reset_paramters(self):
        self.embed.data.normal_()


class NNDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_uni_edge_type, l1_dim=8):
        """ in_dim: the feat dim of a drug
            num_edge_type: num of dd edge type """

        super(NNDecoder, self).__init__()
        self.l1_dim = l1_dim     # Decoder Lays' dim setting

        # parameters
        # for drug 1
        self.w1_l1 = Param(torch.Tensor(in_dim, l1_dim))
        self.w1_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified
        # for drug 2
        self.w2_l1 = Param(torch.Tensor(in_dim, l1_dim))
        self.w2_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type):
        # layer 1
        d1 = torch.matmul(z[edge_index[0]], self.w1_l1)
        d2 = torch.matmul(z[edge_index[1]], self.w2_l1)
        d1 = F.relu(d1, inplace=True)
        d2 = F.relu(d2, inplace=True)

        # layer 2
        d1 = (d1 * self.w1_l2[edge_type]).sum(dim=1)
        d2 = (d2 * self.w2_l2[edge_type]).sum(dim=1)

        return torch.sigmoid(d1 + d2)

    def reset_parameters(self):
        self.w1_l1.data.normal_()
        self.w2_l1.data.normal_()
        self.w1_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))
        self.w2_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))

##### CHECK WHETHER BELOW CODE IS REALLY NECESSARY 
# data.d_feat.requires_grad = True

# n_base = 16

# n_embed = 64
# n_hid1 = 32
# n_hid2 = 16

# encoder = Encoder(n_feat_d, n_et_dd, n_base)
# decoder = NNDecoder(n_hid2, n_et_dd)
# model = MyGAE(encoder, decoder)

# device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device_name)
# device = torch.device(device_name)

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

    z = model.encoder(data.d_feat, data.train_idx, data.train_et, data.train_range, data.x_norm)
    
    pos_score = model.decoder(z, data.test_idx, data.test_et)
    
    return pos_score

print('evaluating over the entire dataset and generating the results...')
with torch.no_grad():
  result = evaluate()
torch.cuda.empty_cache()

print('storing the results...')
old_data = []
if iteration > 1:
  with open('output/evaluation/ddm-nn.pkl', 'rb') as f:
    old_data = pickle.load(f)
with open('output/evaluation/ddm-nn.pkl', 'wb') as f:
  pickle.dump(old_data + result.tolist(), f)
print('Result of this iteration:', result)
print('Total number of records processed till now:', len(old_data + result.tolist()))

# import numpy as np 
# import pickle
# with open('output/evaluation/ddm-nn.pkl', 'rb') as f:
#      a = np.array([pickle.load(f)])
# print(a)