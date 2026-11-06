"""Exploratory contract-wise adapters of PyGOD node detectors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pygod.nn import CoLABase, DOMINANTBase, OCGNNBase
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT/'projects/tds'
SEED = 321
EPOCHS = 5
BATCH = 64
WIDTH = 32


def score(arm, model, batch):
    x, edges, assignment = batch.x, batch.edge_index, batch.batch
    if arm == 'DOMINANT':
        embedding = model.shared_encoder(x, edges)
        reconstructed = model.attr_decoder(embedding, edges)
        structure, mask = to_dense_batch(model.struct_decoder.nn(embedding, edges), assignment)
        adjacency = to_dense_adj(edges, assignment, max_num_nodes=structure.shape[1]).clamp(max=1)
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        errors = ((structure @ structure.transpose(1,2) - adjacency).square()*valid).sum(-1)
        attr = (reconstructed - x).square().sum(-1)
        node_score = .5*(attr+1e-8).sqrt() + .5*(errors[mask]+1e-8).sqrt()
        return node_score.mean(), node_score
    if arm == 'CoLA':
        embedding = model.encoder(x, edges)
        # Shift features by one position within each contract to avoid coupling
        # predictions across held-out contracts. Singleton shifts are identity.
        indices = torch.arange(len(x), device=x.device)
        end = torch.cat((assignment[:-1]!=assignment[1:], assignment.new_ones(1).bool()))
        negative = torch.where(end, batch.ptr[assignment], indices+1)
        pos = model.discriminator(x, embedding).reshape(-1)
        neg = model.discriminator(x[negative], embedding).reshape(-1)
        loss = (torch.nn.functional.softplus(-pos) + torch.nn.functional.softplus(neg)).mean()
        return loss, neg-pos
    embedding = model(x, edges)
    return model.loss_func(embedding)


def run(cohort, arm, device, out):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    path = ROOT/'archive/tds/data/gog_scimain_v1/graph.pt' if cohort=='original' else PROJECT/'private/317/horizon7_graphs.pt'
    population = pd.read_csv(PROJECT/f'results/317/{cohort}_population.csv')
    payload = torch.load(path, map_location='cpu', weights_only=False)['graphs']
    assert len(payload) == len(population)
    graphs = []
    for row, graph in zip(population.itertuples(index=False), payload):
        if cohort=='original':
            x, edges = graph['x'], graph['edge_index']
            identity = (graph['label'], graph['chain_index'], graph['timestamp'])
        else:
            x, edges = graph.x, graph.edge_index
            identity = (graph.y, graph.chain_id, graph.timestamp)
        assert tuple(map(int,identity)) == (row.label,row.chain_index,row.timestamp)
        graphs.append(Data(x=x.float(),edge_index=edges.long()))
    train = [g for g,split in zip(graphs,population.split) if split=='train']
    test = [g for g,split in zip(graphs,population.split) if split=='test']
    models = {'DOMINANT':DOMINANTBase, 'CoLA':CoLABase, 'OCGNN':OCGNNBase}
    model = models[arm](in_dim=3,hid_dim=WIDTH,num_layers=2).to(device)
    if arm=='OCGNN':
        # Estimate and freeze hypersphere statistics from train inputs.
        model.eval()
        with torch.no_grad():
            sample = next(iter(DataLoader(train,batch_size=256,shuffle=True))).to(device)
            embedding = model(sample.x,sample.edge_index)
            model.c = embedding.mean(0)
            model.r = torch.quantile((embedding-model.c).norm(dim=1),1-model.beta)
        model.warmup = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in range(EPOCHS):
        model.train()
        for batch in DataLoader(train,batch_size=BATCH,shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss,_ = score(arm,model,batch)
            loss.backward()
            optimizer.step()
        print(cohort,arm,epoch+1,float(loss.detach()),flush=True)
    values = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(test,batch_size=BATCH):
            batch=batch.to(device)
            _,nodes=score(arm,model,batch)
            values.extend(global_mean_pool(nodes.unsqueeze(-1),batch.batch).flatten().cpu().tolist())
    labels = population.loc[population.split=='test','label'].to_numpy()
    values = np.asarray(values)
    assert len(labels)==len(values) and np.isfinite(values).all()
    order = np.argsort(-values,kind='stable')
    result = dict(cohort=cohort, detector=arm, n=len(labels), positive=int(labels.sum()),
                  ap=float(average_precision_score(labels,values)), roc_auc=float(roc_auc_score(labels,values)),
                  **{f'capture_{k}':int(labels[order[:k]].sum()) for k in (37,50,100)},
                  seed=SEED, epochs=EPOCHS, batch_size=BATCH, hidden=WIDTH,
                  peak_mib=round(torch.cuda.max_memory_allocated(device)/2**20,2))
    (out/f'{cohort}_{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',choices=('original','horizon7'))
    parser.add_argument('arm',choices=('DOMINANT','CoLA','OCGNN'))
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.cohort,args.arm,torch.device(args.device),args.out)
