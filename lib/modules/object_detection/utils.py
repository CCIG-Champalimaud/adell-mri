import torch

resnet_default = [
    (32,64,5,2),(64,128,3,2),
    (128,256,3,2),(256,512,3,1)]
maxpool_default = [
    (2,2,2),(1,1,1),(2,2,1),(2,2,1)]
pyramid_default = [3,5,[7,7,5],[9,9,5],[11,11,5]]

def check_overlap(bb1:torch.Tensor,bb2:torch.Tensor,ndim:int=3)->torch.Tensor:
    return torch.logical_and(
        torch.any(bb1[:,ndim:] > bb2[:,:ndim],axis=1),
        torch.any(bb1[:,:ndim] < bb2[:,ndim:],axis=1))

def bb_volume(bb:torch.Tensor,ndim:int=3)->torch.Tensor:
    return torch.prod(bb[:,ndim:] - bb[:,:ndim]+1,axis=1)

def calculate_iou(bb1:torch.Tensor,bb2:torch.Tensor,ndim:int=3)->torch.Tensor:
    inter_tl = torch.maximum(
        bb1[:,:ndim],bb2[:,:ndim])
    inter_br = torch.minimum(
        bb1[:,ndim:],bb2[:,ndim:])
    inter_volume = torch.prod(inter_br - inter_tl + 1,axis=1)
    union_volume = bb_volume(bb1,ndim)+bb_volume(bb2,ndim)-inter_volume
    return inter_volume/union_volume

def nms_nd(bb:torch.Tensor,scores:torch.Tensor,
           score_threshold:float,iou_threshold:float=0.5)->torch.Tensor:
    # first we sort the scores and boxes according to the scores and 
    # remove boxes with a score below `score_threshold`
    n,ndim = bb.shape
    ndim = int(ndim//2)
    original_idx = torch.arange(n,dtype=torch.long)
    scores_idxs = scores > score_threshold
    scores,bb = scores[scores_idxs],bb[scores_idxs]
    original_idx = original_idx[scores_idxs]
    score_order = torch.argsort(scores).flip(0)
    scores = scores[score_order]
    bb = bb[score_order]
    original_idx = original_idx[score_order]
    excluded = torch.zeros_like(scores,dtype=bool)
    idxs = torch.arange(scores.shape[0],dtype=torch.long)
    # iteratively remove boxes which have a high overlap with other boxes,
    # keeping those with higher confidence
    for i in range(bb.shape[0]):
        if excluded[i] is False:
            cur_bb = torch.unsqueeze(bb[i],0)
            cur_excluded = excluded[(i+1):]
            cur_idxs = idxs[(i+1):][~cur_excluded]
            remaining_bb = bb[cur_idxs]
            overlap = check_overlap(cur_bb,remaining_bb,ndim)
            remaining_bb = remaining_bb[overlap]
            cur_idxs = cur_idxs[overlap]
            iou = calculate_iou(cur_bb,remaining_bb,ndim)
            cur_idxs = cur_idxs[iou>iou_threshold]
            if cur_idxs.shape[0] > 0:
                excluded[cur_idxs] = True
    return original_idx[~excluded]