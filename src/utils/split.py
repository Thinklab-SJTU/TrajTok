import torch

def split_by_type(data: torch.Tensor, type_mask: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Split a tensor by type mask.
    """
    if not isinstance(data, dict):
        res = {k: data[mask] for k, mask in type_mask.items()}
    else:
        res = {}
        for agent_type, mask in type_mask.items():
            res[agent_type] = {}
            for data_k, v in data.items():
                res[agent_type][data_k] = v[mask]
    return res

    de