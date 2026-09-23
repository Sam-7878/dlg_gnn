import multiprocessing as mp
import torch
from gog_fraud.models.pygod.gadnr import GADNRBase, _UnusedPool


def test_gadnr_does_not_start_unused_process_pool():
    before = {child.pid for child in mp.active_children()}
    model = GADNRBase(in_dim=3, hid_dim=4, encoder_layers=1,
                      deg_dec_layers=1, fea_dec_layers=1, sample_size=2,
                      sample_time=1, neighbor_num_list=torch.tensor([0, 1]),
                      neigh_loss="KL", lambda_loss1=.01, lambda_loss2=.1,
                      lambda_loss3=.8, full_batch=True, device=torch.device("cpu"))
    after = {child.pid for child in mp.active_children()}
    assert before == after
    assert isinstance(model.pool, _UnusedPool)
