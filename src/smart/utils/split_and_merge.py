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

def merge_by_type(data: dict[str, dict[str, torch.Tensor]],mask: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Merge a dictionary of tensors by type.
    """
    first_value = list(data.values())[0]
    if isinstance(first_value, torch.Tensor):
        input_shape = {k: v.shape for k,v in data.items()}
        output_shape = list(first_value.shape)
        output_shape[0] = sum([s[0] for s in input_shape.values()])
        res = torch.zeros(output_shape, dtype=first_value.dtype, device=first_value.device)
        for k,v in data.items():
            res[mask[k]] = v
        return res
    
    else:
        res = {}
        
        for data_k, another_dict in data.items():
            res[data_k] = merge_by_type(another_dict,mask)
        return res



