import json

import torch
from torch_geometric.data import Data

from gog_fraud.experiments.round4c_policy import should_skip
from gog_fraud.models.pygod.shared_reconstruction import SharedDOMINANT


def test_resume_skips_only_success_or_objective_unsupported(tmp_path):
    path=tmp_path/"cell.json"
    for status,expected in (("success",True),("unsupported_operational",True),("failed_cuda",False)):
        path.write_text(json.dumps({"status":status}))
        assert should_skip(path,resume=True) is expected
    assert not should_skip(path,resume=False)


def test_exact_training_checkpoint_resumes_at_next_epoch(tmp_path):
    data=Data(x=torch.randn(7,4),edge_index=torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,6]]),num_nodes=7)
    checkpoint=tmp_path/"training.pt"
    first=SharedDOMINANT(epoch=2,gpu=-1,verbose=0)
    first.training_checkpoint_path=checkpoint
    first.fit(data.clone())
    progress=json.loads((tmp_path/"training.pt.progress.json").read_text())
    assert progress["completed_epochs"] == 2
    resumed=SharedDOMINANT(epoch=4,gpu=-1,verbose=0)
    resumed.training_checkpoint_path=checkpoint
    resumed.fit(data.clone())
    assert resumed.resumed_from_epoch_ == 2
    assert resumed.actual_epochs_ == 4
