import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import time
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from Constants import *
from torch.optim import Adam
from dataset import GetFrameData
from torch.utils.data import DataLoader
from model import Build_Graph, PhyNetwork
from bayes_opt import BayesianOptimization
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler as Scaler

parser = argparse.ArgumentParser()
parser.add_argument('--init_lr', type=float, default=0.001) 
parser.add_argument("--epoch", type=int, default=50, help="epoch")
parser.add_argument("--logdir", type=str, default='default', help="logdir")

parser.add_argument("--cutoff_radius",  type=float, default= 0.085, help="cutoff radius when build graph")
parser.add_argument("--radius_cell_ratio", type=float, default= 1.6, help="radius_cell_ratio")
parser.add_argument("--sample_num", type=int, default=1000, help="sample num")
parser.add_argument("--seed_num", type=int, default=5, help="train data dir number")
parser.add_argument("--val_seed_num", type=int, default=1, help="validation data dir number")
parser.add_argument("--train_data", type=str, default='../FGN/dataset/training1', help="train dataset path")
parser.add_argument("--val_data", type=str, default='../FGN/dataset/testing1', help="val dataset path")
args = parser.parse_args()
print("Parameter: ", args)

EPOCHS = args.epoch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#train data
train_data = GetFrameData('../FGN/dataset/training1',args.sample_num, seed_num=args.seed_num)
train_data_num = len(train_data)
train_data_loader = DataLoader(train_data, 1, shuffle=True, num_workers=4)
print(f'Training use {train_data_num} data points in total')

#val data
val_data = GetFrameData(args.val_data, args.sample_num, seed_num=args.val_seed_num)
val_data_num = len(val_data)
val_data_loader = DataLoader(val_data, 1, shuffle=True, num_workers=4)
print(f'Val use {val_data_num} data points in total')

model = PhyNetwork(4, 3, 2, args.cutoff_radius, args.radius_cell_ratio).to(device)

def get_edge_index(vd_graph, pos):
    # build new graph and 
    dists, nbr_idxs, grid = vd_graph.frnn_process(pos, args.radius_cell_ratio)
    g, edge_index = vd_graph.build_fixed_radius_graph(dists, nbr_idxs, args.cutoff_radius)
    return edge_index

def train_per_epoch(epoch, w1, w2, w3, writer, optim):
    model.train()
    
    tot_loss = 0.
    sum_loss = 0.
    sum_loss1 = 0.
    sum_loss2 = 0.
    sum_phyloss = 0.
    pbar = tqdm(enumerate(train_data_loader))
    
    vd_graph = Build_Graph(store_weights=True, cutoff_radius = args.cutoff_radius)
    for current, data in pbar:
        pos = data['pos_prev'].squeeze(0).to(device)
        vel = data['vel_prev'].squeeze(0).to(device)
        vel_after = data['vel_after'].squeeze(0).to(device)
        vel_after_after = data['vel_after_after'].squeeze(0).to(device)
        
        vel_norm = torch.norm(vel, dim=1).view(-1,1) + 1e-8
        vel /= vel_norm
        vel_norm_scaler = Scaler()
        vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
        vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item() + 1e-10)
        x = torch.cat((vel, vel_norm), dim=1)
        # print(np.any(np.isnan(vel_norm.cpu().numpy())))
        pred_1, pred_2 = model(x, pos) #predicte delta vel
        
        loss1 = nn.MSELoss()(pred_1, vel_after)
        loss2 = nn.MSELoss()(pred_2, vel_after_after)
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        
        edge_index = get_edge_index(vd_graph, pre_pos_1)
        phy_loss = velocity_divergence(pre_vel_1, pre_pos_1, edge_index)
        sum_phyloss += phy_loss
        
        tot_loss = w1 * loss1 + w2 * loss2 + w3 * phy_loss
        sum_loss += tot_loss.item()
        
        optim.zero_grad()
        tot_loss.backward()
        optim.step() 
        
    Total_loss = sum_loss/train_data_num
    Loss1 = sum_loss1/train_data_num
    Loss2 = sum_loss2/train_data_num
    Phy_loss = sum_phyloss/train_data_num
    print(Total_loss, Loss1, Loss2, Phy_loss)
    print(f'\nTraining Epoch{epoch}: total loss: {Total_loss}, loss1: {Loss1}, loss2: {Loss2}, Phy loss: {Phy_loss}')
    writer.add_scalar('Train/loss1',      Loss1, epoch)
    writer.add_scalar('Train/loss2',      Loss2, epoch)
    writer.add_scalar('Train/phy_loss',   Phy_loss, epoch)
    writer.add_scalar('Train/total_loss', Total_loss, epoch)        

          
def train_evaluate(w1,w2,w3):
    os.makedirs('./weilan/record/{}_{}_{}_{}'.format(args.logdir,'%.2f'%w1, '%.2f'%w2, '%.2f'%w3), exist_ok=True)
    os.makedirs('./weilan/record/{}_{}_{}_{}/model'.format(args.logdir,'%.2f'%w1, '%.2f'%w2, '%.2f'%w3), exist_ok=True)
    writer = SummaryWriter('./weilan/record/{}_{}_{}_{}/log'.format(args.logdir,'%.2f'%w1, '%.2f'%w2, '%.2f'%w3))
    best_path = './weilan/record/{}_{}_{}_{}/model/best.pkl'.format(args.logdir,'%.2f'%w1, '%.2f'%w2, '%.2f'%w3)
    latest_path = './weilan/record/{}_{}_{}_{}/model/latest.pkl'.format(args.logdir,'%.2f'%w1, '%.2f'%w2, '%.2f'%w3)
    
    min_chamfer_dist = 10
    print(f'w1: {w1}, w2: {w2}, w3:{w3}')

    optim = Adam([{"params": model.parameters(), 'lr': args.init_lr}])
    scheduler = StepLR(optim, 3, gamma=0.333, last_epoch=-1)   
          
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_per_epoch(epoch, w1, w2, w3, writer, optim)
        chamfer_dist1, chamfer_dist2 = evaluation(epoch, val_data_loader, writer)
        scheduler.step()
          
        if chamfer_dist1 < min_chamfer_dist:
            best_epoch = epoch
            min_chamfer_dist = chamfer_dist1
            torch.save(model.state_dict(), best_path)
          
        torch.save(model.state_dict(), latest_path)
        
        time_elapsed = time.time() - start
        print('Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
    return -min_chamfer_dist
          
        
optimizer = BayesianOptimization(
    train_evaluate,
    {'w1': (0, 1),
    'w2': (0, 1),
    'w3': (0, 1)})

optimizer.maximize(n_iter=10)
print("Final result:", optimizer.max) 
        
        
        
        
        
        
        
        
        
        
        
        
