from gog_fraud.experiments.ablation_variants import get_ablation_variant


def test_dlg_full_exports_empirical_local_score():
    import torch
    from torch_geometric.data import Data
    from gog_fraud.models.pygod.dlg_full import DLGFull

    nodes = 20
    data = Data(
        x=torch.randn(nodes, 4),
        edge_index=torch.stack([torch.arange(nodes), torch.roll(torch.arange(nodes), -1)]),
        y=torch.tensor([0] * 16 + [1] * 4),
        num_nodes=nodes,
    )
    model = DLGFull(epoch=1, l1_epochs=1, gpu=-1, batch_size=0, verbose=0)
    model.fit(data)
    assert data.dlg_l1_score.shape == (nodes,)
    assert torch.isfinite(data.dlg_l1_score).all()


def test_required_variants_have_distinct_component_flags():
    l1 = get_ablation_variant("l1_only")
    l1_l2 = get_ablation_variant("l1_l2")
    full = get_ablation_variant("full")
    assert (l1.use_local, l1.use_global, l1.use_fusion) == (True, False, False)
    assert (l1_l2.use_local, l1_l2.use_global, l1_l2.use_fusion) == (True, True, False)
    assert (full.use_local, full.use_global, full.use_fusion) == (True, True, True)
