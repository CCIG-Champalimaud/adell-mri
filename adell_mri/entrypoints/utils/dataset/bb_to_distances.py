
desc = "Calculates the minimum distance between bounding \
            boxes in the same image."


def main(arguments):
    import argparse

    import numpy as np
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--input_path",
        dest="input_path",
        required=True,
        help="Path to csv with bounding boxes",
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
        "--min_size",
        dest="min_size",
        default=0,
        type=int,
        help="Minimum number of input pixels per output pixel",
    )
    args = parser.parse_args(arguments)

    all_bb = {}
    all_shapes = {}
    with open(args.input_path, "r") as o:
        for line in o:
            line = line.strip().split(",")
            image_id = line[0]
            bb = np.array(line[1 : -args.spatial_dim - 1]).astype(np.int32)
            sh = np.array(line[-args.spatial_dim - 1 : -1]).astype(np.int32)
            if image_id in all_bb:
                all_bb[image_id].append(bb)
            else:
                all_bb[image_id] = [bb]
            all_shapes[image_id] = sh

    all_distances = []
    all_shapes_w_distance = []
    keys_w_distance = []
    for image_id in all_bb:
        arr = np.array(all_bb[image_id]).astype(np.float32)
        if arr.shape[0] > 1:
            sh = all_shapes[image_id]
            all_dists = []
            for i in range(arr.shape[0]):
                for j in range(arr.shape[0]):
                    if i != j:
                        c1 = (
                            arr[i, : args.spatial_dim]
                            + arr[i, args.spatial_dim :]
                        )
                        c2 = (
                            arr[j, : args.spatial_dim]
                            + arr[j, args.spatial_dim :]
                        )
                        c1 /= 2
                        c2 /= 2
                        all_dists.append(np.abs(c1 - c2))
            d = np.array(all_dists)
            d = d[d.sum(1).argmin()]
            if args.min_size > 0:
                if np.all(d > (sh / args.min_size)):
                    all_distances.append(d)
                    all_shapes_w_distance.append(sh)
                    keys_w_distance.append(image_id)
            else:
                all_distances.append(d)
                all_shapes_w_distance.append(sh)
                keys_w_distance.append(image_id)

    all_distances = np.array(all_distances)
    all_shapes_w_distance = np.array(all_shapes_w_distance)
    am = np.argmin(np.sum(all_distances, axis=1))
    sh = all_shapes_w_distance[am]
    min_dist = np.floor(all_distances[am])
    print(min_dist)
