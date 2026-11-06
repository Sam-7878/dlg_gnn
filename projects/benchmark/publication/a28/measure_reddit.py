"""Measure disjoint full-graph DLG-Aug fit phases without retaining scores."""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from a24_inputs import load_input
from a24_common import sha
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGFull

def tick():
    torch.cuda.synchronize()
    return time.perf_counter()

class TimedDLG(SharedDLGFull):
    def process_graph(self, data):
        if getattr(data, "_dlg_full_augmented", False):
            return super().process_graph(data)
        started = tick()
        result = super().process_graph(data)
        self.graph_prepare_and_local_seconds = tick() - started
        return result

    def _pretrain_level1(self, data):
        started = tick()
        result = super()._pretrain_level1(data)
        self.local_pretraining_seconds = tick() - started
        return result

def main(output):
    assert '3090' in torch.cuda.get_device_name(0)
    start=time.perf_counter()
    graph, _, identity=load_input('Reddit-Syn')
    data=Data(x=graph.x.clone(),edge_index=graph.edge_index.clone(),num_nodes=graph.num_nodes)
    input_load_and_clone_seconds=time.perf_counter()-start
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    model=TimedDLG(gpu=0,hid_dim=64,num_layers=4,lr=.004,dropout=0.,weight_decay=0.,
                   batch_size=0,verbose=0,epoch=50,l1_epochs=20,l1_hid_dim=64,l1_hops=2)
    torch.cuda.reset_peak_memory_stats()
    started=tick()
    model.fit(data)
    fitted=tick()
    scores=model.decision_function(data)
    scored=tick()
    assert model.actual_epochs_==50 and model.l1_epochs==20
    assert scores.numel()==data.num_nodes and torch.isfinite(scores).all()
    phases={
        'fit_preprocessing_seconds':model.graph_prepare_and_local_seconds-model.local_pretraining_seconds,
        'local_pretraining_seconds':model.local_pretraining_seconds,
        'global_fit_including_setup_and_finalization_seconds':fitted-started-model.graph_prepare_and_local_seconds,
    }
    assert min(phases.values())>0
    fit=fitted-started
    assert abs(sum(phases.values())-fit)<1e-8
    record=dict(dataset='Reddit-Syn',model='DLG-Aug',seed=seed,local_epochs=20,global_epochs=50,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                tensor_sha256=identity['source_sha256'],instrumentation_sha256=sha(__file__),
                input_load_and_clone_seconds=input_load_and_clone_seconds,**phases,fit_seconds=fit,
                score_seconds=scored-fitted,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                local_scope='Includes local message graph setup, 20 epochs and extraction of embeddings.',
                global_scope='Includes global setup, 50 epochs and fit finalization.')
    tmp=output.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2)+'\n')
    tmp.replace(output)
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    main(Path(sys.argv[1]))
