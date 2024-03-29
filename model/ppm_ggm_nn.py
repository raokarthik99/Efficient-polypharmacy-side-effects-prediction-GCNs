from data.utils import load_data_torch
import pickle
from torch.nn import Module
from src.utils import *
from src.layers import *
import sys
import time
import os
import matplotlib.pyplot as plt


sys.setrecursionlimit(8000)

with open('data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = 'output/ppm-ggm-nn/'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

EPOCH_NUM = 100

#########################################################################
#et_list = et_list[:10]       # remove this line for full dataset learning
#########################################################################

# et_list = et_list
feed_dict = load_data_torch("data/", et_list, mono=True)
data = Data.from_dict(feed_dict)
n_drug, n_drug_feat = data.d_feat.shape
n_prot, n_prot_feat = data.p_feat.shape
n_et_dd = len(et_list)

data.train_idx, data.train_et, data.train_range, data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index)

# re-construct node feature
data.p_feat = torch.cat([dense_id(n_prot), torch.zeros(size=(n_drug, n_prot))], dim=0)
data.d_feat = dense_id(n_drug)
n_drug_feat = n_drug
n_prot_feat = n_prot

# ###################################
# dp_edge_index and range index
# ###################################
data.dp_edge_index = np.array([data.dp_adj.col-1, data.dp_adj.row-1])

count_durg = np.zeros(n_drug, dtype=np.int)
for i in data.dp_edge_index[1, :]:
    count_durg[i] += 1
range_list = []
start = 0
end = 0
for i in count_durg:
    end += i
    range_list.append((start, end))
    start = end

data.dp_edge_index = torch.from_numpy(data.dp_edge_index + np.array([[0], [n_prot]]))
data.dp_range_list = range_list


data.d_norm = torch.ones(n_drug)
data.p_norm = torch.ones(n_prot+n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))
# data.d_feat.requires_grad = True


source_dim = n_prot_feat
embed_dim = 32
target_dim = 16


class HierEncoder(Module):
    def __init__(self, source_dim, embed_dim, target_dim,
                 uni_num_source, uni_num_target):
        super(HierEncoder, self).__init__()

        self.embed = Param(torch.Tensor(source_dim, embed_dim))
        self.hgcn = MyHierarchyConv(embed_dim, target_dim, uni_num_source, uni_num_target)

        self.reset_parameters()

    def reset_parameters(self):
        self.embed.data.normal_()

    def forward(self, source_feat, edge_index, range_list, x_norm):
        x = torch.matmul(source_feat, self.embed)
        x = x / x_norm.view(-1, 1)
        x = self.hgcn(x, edge_index, range_list)
        # x = F.relu(x, inplace=True)

        return x


# class NNDecoder(Module):
#     def __init__(self, in_dim, num_uni_edge_type, l1_dim=16):
#         """ in_dim: the feat dim of a drug
#             num_edge_type: num of dd edge type """
#
#         super(NNDecoder, self).__init__()
#         self.l1_dim = l1_dim     # Decoder Lays' dim setting
#
#         # parameters
#         # for drug 1
#         self.w1_l1 = Param(torch.Tensor(in_dim, l1_dim))
#         self.w1_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
#         # specified
#         # for drug 2
#         self.w2_l1 = Param(torch.Tensor(in_dim, l1_dim))
#         self.w2_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
#         # specified
#
#         self.reset_parameters()
#
#     def forward(self, z, edge_index, edge_type):
#         # layer 1
#         d1 = torch.matmul(z[edge_index[0]], self.w1_l1)
#         d2 = torch.matmul(z[edge_index[1]], self.w2_l1)
#         d1 = F.relu(d1, inplace=True)
#         d2 = F.relu(d2, inplace=True)
#
#         # layer 2
#         d1 = (d1 * self.w1_l2[edge_type]).sum(dim=1)
#         d2 = (d2 * self.w2_l2[edge_type]).sum(dim=1)
#
#         return torch.sigmoid(d1 + d2)
#
#     def reset_parameters(self):
#         self.w1_l1.data.normal_()
#         self.w2_l1.data.normal_()
#         self.w1_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))
#         self.w2_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))


encoder = HierEncoder(source_dim, embed_dim, target_dim, n_prot, n_drug)
decoder = NNDecoder(target_dim, n_et_dd)
model = MyGAE(encoder, decoder)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_name = 'cpu'
print(device_name)
device = torch.device(device_name)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
data = data.to(device)

train_out = {}
train_record = {}
test_out = {}
test_record = {}

@profile
def train():

    model.train()

    optimizer.zero_grad()

    z = model.encoder(data.p_feat, data.dp_edge_index, data.dp_range_list, data.p_norm)

    pos_index = data.train_idx
    neg_index = negative_sampling(data.train_idx, n_drug).to(device)

    pos_score = model.decoder(z, pos_index, data.train_et)
    neg_score = model.decoder(z, neg_index, data.train_et)

    # pos_loss = F.binary_cross_entropy(pos_score, torch.ones(pos_score.shape[0]).cuda())
    # neg_loss = F.binary_cross_entropy(neg_score, torch.ones(neg_score.shape[0]).cuda())
    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss
    # loss = pos_loss

    loss.backward()
    optimizer.step()

    record = np.zeros((3, n_et_dd))  # auprc, auroc, ap
    for i in range(data.train_range.shape[0]):
        [start, end] = data.train_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    train_record[epoch] = record
    [auprc, auroc, ap] = record.sum(axis=1) / n_et_dd
    train_out[epoch] = [auprc, auroc, ap]
    
    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}'
          .format(epoch, loss.tolist(), auprc, auroc, ap))

    return z, loss


test_neg_index = negative_sampling(data.test_idx, n_drug).to(device)


def test(z):
    model.eval()

    record = np.zeros((3, n_et_dd))     # auprc, auroc, ap

    pos_score = model.decoder(z, data.test_idx, data.test_et)
    neg_score = model.decoder(z, test_neg_index, data.test_et)

    for i in range(data.test_range.shape[0]):
        [start, end] = data.test_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    return record

print('model training ...')
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train()

    auprc, auroc, ap = test(z)
    record_te = test(z)
    [auprc, auroc, ap] = record_te.sum(axis=1) / n_et_dd

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.1f}\n'
          .format(epoch, loss.tolist(), auprc, auroc, ap, (time.time() - time_begin)))

    test_record[epoch] = record_te
    test_out[epoch] = [auprc, auroc, ap]

# # save output to files
with open(out_dir + 'train_out.pkl', 'wb') as f:
    pickle.dump(train_out, f)

with open(out_dir + 'test_out.pkl', 'wb') as f:
    pickle.dump(test_out, f)
    
with open(out_dir + 'train_record.pkl', 'wb') as f:
    pickle.dump(train_record, f)

with open(out_dir + 'test_record.pkl', 'wb') as f:
    pickle.dump(test_record, f)

# save model state
filepath_state = os.path.join(out_dir, '100ep.pth')
torch.save(model.state_dict(), filepath_state)
# to restore
model.load_state_dict(torch.load(filepath_state))
model.eval()

# save whole model
filepath_model = os.path.join(out_dir, '100ep_model.pb')
torch.save(model, filepath_model)
# Then later:
model = torch.load(filepath_model)


# ##################################
# training and testing figure
def dict_to_nparray(out_dict, epoch):
    out = np.zeros(shape=(3, epoch))
    for ep, [prc, roc, ap] in out_dict.items():
        out[0, ep] = prc
        out[1, ep] = roc
        out[2, ep] = ap
    return out


tr_out = dict_to_nparray(train_out, EPOCH_NUM)
te_out = dict_to_nparray(test_out, EPOCH_NUM)

plt.figure()
x = np.array(range(EPOCH_NUM), dtype=int) + 1
maxmum = np.zeros(EPOCH_NUM) + te_out[0, :].max()
plt.plot(x, tr_out[0, :], label='train_prc')
plt.plot(x, te_out[0, :], label='test_prc')
plt.plot(x, maxmum, linestyle="-.")
plt.title('AUPRC scores - PPM_GGM_NN')
plt.grid()
plt.legend()
plt.savefig(out_dir + 'prc.png')
plt.show()
