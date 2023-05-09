import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from ..custom_types import TensorOrNDarray
from typing import List,Dict

def anchors_from_nested_list(nested_list:List[Dict[str,TensorOrNDarray]],
                             bounding_box_key:str,
                             shape_key:str,
                             iou_threshold:float=0.5):
    
    all_bb = []
    for i,x in enumerate(nested_list):
        if bounding_box_key in x and shape_key in x:
            bb = np.array(x[bounding_box_key])
            for i in range(len(bb)):
                all_bb.append(bb[i])

    all_bb = np.array(all_bb)
    ndim = all_bb.shape[1] // 2
    all_sizes = all_bb[:,ndim:] - all_bb[:,:ndim] + 1
    all_bb = np.concatenate([-all_sizes/2,all_sizes/2],1)
    all_areas = np.prod(all_sizes,1)
        
    up = tqdm()
    cont = True
    i = 2
    up.set_description("Determining the ideal anchor size")
    while cont == True:
        up.update()
        km = KMeans(i,random_state=42)
        km = km.fit(all_sizes)
        centers = km.cluster_centers_
        iou = []
        center_corners = np.concatenate(
            [-centers/2,centers/2],1)
        good = np.zeros([all_bb.shape[0]])
        for center,center_corner in zip(centers,center_corners):
            inter_tl = np.maximum(
                center_corner[:ndim],all_bb[:,:ndim])
            inter_br = np.minimum(
                center_corner[ndim:],all_bb[:,ndim:])
            inter = np.prod(inter_br - inter_tl + 1,axis=1)
            union = all_areas + np.prod(center + 1) - inter
            iou = inter/union
            good[iou>iou_threshold] += 1
        
        if np.sum(good > 0) >= all_bb.shape[0]:
            cont = False
        i += 1

    cluster_assignment = km.predict(all_sizes)
    anchors = []
    for k in np.unique(cluster_assignment):
        c = np.mean(all_sizes[cluster_assignment == k],axis=0)
        anchors.append(c)
    anchors = np.array(anchors)
    print("Inferred {} anchors:".format(anchors.shape[0]))
    for i in range(anchors.shape[0]):
        print("\tAnchor {}: {}".format(i,anchors[i]))
    return anchors
