import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates the minimum distance between bounding \
            boxes in the same image."
    )

    parser.add_argument(
        "--input_path",
        dest="input_path",
        required=True,
        help="Path to JSON with bounding boxes",
    )
    parser.add_argument(
        "--spatial_dim",
        dest="spatial_dim",
        default=2,
        type=int,
        choices=[2, 3],
        help="Spatial dimension",
    )
    parser.add_argument(
        "--seed", dest="seed", default=42, type=int, help="Seed for clustering"
    )
    parser.add_argument(
        "--iou_threshold",
        dest="iou_threshold",
        default=0.25,
        type=float,
        help="IoU threshold for considering an object to be detected",
    )
    args = parser.parse_args()

    all_bb = []
    all_sh = []
    all_ids = []
    ndim = args.spatial_dim
    with open(args.input_path, "r") as o:
        data = json.load(o)

    for k in data:
        box = data[k]["boxes"][0]
        shape = data[k]["shape"]
        image_id = k
        bb = np.array(box).astype(np.int32)
        sh = np.array(shape).astype(np.int32)
        bb = bb[ndim:] - bb[:ndim]
        if np.all(bb > 0):
            all_bb.append(bb)
            all_sh.append(sh)
            all_ids.append(image_id)

    all_bb = np.array(all_bb)
    all_sh = np.array(all_sh)
    median_sh = np.median(all_sh, axis=0)
    sh_adj = all_sh / median_sh
    d = all_bb / sh_adj
    all_corners = np.concatenate([-d / 2, d / 2], 1)
    all_areas = np.prod(d + 1, 1)

    up = tqdm()
    cont = True
    i = 2
    while cont == True:
        up.update()
        km = KMeans(i, random_state=42)
        km = km.fit(d)
        centers = km.cluster_centers_
        iou = []
        center_corners = np.concatenate([-centers / 2, centers / 2], 1)
        good = np.zeros([all_bb.shape[0]])
        for center, center_corner in zip(centers, center_corners):
            inter_tl = np.maximum(center_corner[:ndim], all_corners[:, :ndim])
            inter_br = np.minimum(center_corner[ndim:], all_corners[:, ndim:])
            inter = np.prod(inter_br - inter_tl + 1, axis=1)
            union = all_areas + np.prod(center + 1) - inter
            iou = inter / union
            good[iou > args.iou_threshold] += 1

        if np.sum(good > 0) >= all_bb.shape[0]:
            cont = False
        i += 1

    cluster_assignment = km.predict(d)

    for k in np.unique(cluster_assignment):
        c = np.mean(d[cluster_assignment == k], axis=0)
        print(",".join([str(x) for x in c]))
