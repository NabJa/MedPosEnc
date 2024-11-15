import torch
from einops import rearrange, reduce, repeat


def get_attention_map_from_transfomer_block(x, block):
    x = block.mlp(x)
    x = block.norm1(x)
    x = block.attn.input_rearrange(block.attn.qkv(x))
    q, k = x[0], x[1]
    att_mat = torch.einsum("blxd,blyd->blxy", q, k) * block.attn.scale
    return att_mat.softmax(dim=-1)


def find_module(model, module_name):
    for name, module in model.named_modules():
        if module_name in name:
            return module
    return None


def get_attention_maps(model, img, dataset):
    attn_maps = []

    with torch.no_grad():
        seq = find_module(model, "patch_embedding")(img.unsqueeze(0))
        for block in find_module(model, "blocks"):
            attn_maps.append(get_attention_map_from_transfomer_block(seq, block))
            seq = block(seq)

    attn_maps = rearrange_to_dataset(attn_maps, dataset)

    return attn_maps[11].sum(0)


def rearrange_to_dataset(attn_maps, dataset):
    attn_maps = torch.stack(attn_maps)
    attn_maps = reduce(attn_maps, "depth () head h w -> depth head w", "sum")

    if dataset == "brats":
        attn_maps = rearrange(
            attn_maps, "depth head (a b c) -> depth head a b c", a=6, b=7, c=6
        )
        return repeat(
            attn_maps,
            "depth head a b c -> depth head (a t1) (b t2) (c t3)",
            t1=16,
            t2=16,
            t3=16,
        )
    elif dataset == "echonet":
        attn_maps = rearrange(
            attn_maps, "depth head (a b c) -> depth head a b c", a=16, b=7, c=7
        )
        return repeat(
            attn_maps,
            "depth head a b c -> depth head (a t1) (b t2) (c t3)",
            t1=1,
            t2=16,
            t3=16,
        )
    elif dataset == "chestx":
        attn_maps = rearrange(attn_maps, "depth head (w k) -> depth head w k", k=14)
        return repeat(
            attn_maps, "depth head w k -> depth head (w t1) (k t2)", t1=16, t2=16
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")
