"""
Based on https://github.com/lRomul/argus-tgs-salt/blob/master/mosaic/create_mosaic.py
"""
import argparse
import os
import os.path as osp
from collections import defaultdict
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
from tqdm import tqdm

SOURCE2SHAPE = {
    "arvalis_1": (2, 3),
    "arvalis_2": (2, 2),
    "arvalis_3": (2, 2),
    "inrae_1": (2, 2),
    "usask_1": (1, 2),
    "rres_1": (2, 3),
    "ethz_1": (1, 2),
}
IMG_SHAPE = (1024, 1024)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances-pattern", default="/data/distances/*.npz")
    return parser.parse_args()


def coords_for_open_list(i, j, neighbours, open_list, close_list):
    to_open_list = []
    for n, k in enumerate(neighbours):
        if k is not None and k not in close_list:
            close_list.append(k)
            i_k = i
            j_k = j
            if n == 0:
                i_k -= 1  # left
            elif n == 1:
                j_k -= 1  # top
            elif n == 2:
                i_k += 1  # right
            else:
                j_k += 1  # bottom
            to_open_list.append((i_k, j_k, k))
    return open_list + to_open_list, close_list


def create_mosaic_dict(images_paths, tiles, tile_to_cluster):
    tiles_dicts = {}
    for i in range(len(images_paths)):
        tiles_dicts[i] = {"path": images_paths[i], "cluster": tile_to_cluster[tiles[i][0]], "neighbours": tiles[i][1:]}
    return tiles_dicts


def create_mosaics(tiles_dicts):
    def coords_ij(ij, lim):
        i, j = ij
        i_max, j_max, i_min, j_min = lim
        x_min = (i - i_min) * IMG_SHAPE[1]
        x_max = (i + 1 - i_min) * IMG_SHAPE[1]
        y_min = (j - j_min) * IMG_SHAPE[0]
        y_max = (j + 1 - j_min) * IMG_SHAPE[0]
        return x_min, x_max, y_min, y_max

    def add_img(mosaic, lim, el):
        i_max, j_max, i_min, j_min = lim
        tile = tiles_dicts[el[2]]
        if el[1] < j_min:
            mosaic = np.pad(mosaic, ((IMG_SHAPE[0], 0), (0, 0), (0, 0)), mode="constant")
            j_min -= 1
        if el[1] > j_max:
            mosaic = np.pad(mosaic, ((0, IMG_SHAPE[0]), (0, 0), (0, 0)), mode="constant")
            j_max += 1
        if el[0] < i_min:
            mosaic = np.pad(mosaic, ((0, 0), (IMG_SHAPE[0], 0), (0, 0)), mode="constant")
            i_min -= 1
        if el[0] > i_max:
            mosaic = np.pad(mosaic, ((0, 0), (0, IMG_SHAPE[0]), (0, 0)), mode="constant")
            i_max += 1
        # print('res mosaic', mosaic.shape)
        x_min, x_max, y_min, y_max = coords_ij(el[:2], (i_max, j_max, i_min, j_min))
        # print("Add", el[:2], el[2], [x_min, x_max, y_min, y_max])
        mosaic[y_min:y_max, x_min:x_max] = cv2.imread(tile["path"])
        return mosaic, i_max, j_max, i_min, j_min

    mosaics = []
    dicts_to_save = []
    while len(tiles_dicts) > 0:
        min_idx = min(tiles_dicts.keys())
        tile = tiles_dicts[min_idx]
        del tiles_dicts[min_idx]
        mosaics.append(cv2.imread(tile["path"]))
        i = 1
        j = 1
        i_max = 1
        j_max = 1
        i_min = 1
        j_min = 1
        open_list = []
        close_list = [min_idx]
        open_list, close_list = coords_for_open_list(i, j, tile["neighbours"], open_list, close_list)
        dicts_to_save.append([])
        dicts_to_save[-1].append({"i": i, "j": j, "path": tile["path"]})
        while len(open_list) > 0:
            el = open_list.pop(0)
            if el[2] in tiles_dicts.keys():
                mosaics[-1], i_max, j_max, i_min, j_min = add_img(mosaics[-1], (i_max, j_max, i_min, j_min), el)
                i, j = el[:2]
                open_list, close_list = coords_for_open_list(
                    i, j, tiles_dicts[el[2]]["neighbours"], open_list, close_list
                )
                dicts_to_save[-1].append({"i": i, "j": j, "path": tiles_dicts[el[2]]["path"]})
                del tiles_dicts[el[2]]
                # print(open_list, close_list)
    return mosaics, dicts_to_save


def run(stage, source, left_matrix, top_matrix, image_paths, threshold):
    n_samples = len(image_paths)
    tiles = [[i, None, None, None, None] for i in range(n_samples)]

    print("Create left neighbours")
    left_thres = np.median(left_matrix, axis=1) / threshold
    left_idx = np.argsort(left_matrix, axis=1)[:, 0]

    for i in range(n_samples):
        j = int(left_idx[i])
        if left_matrix[i, j] < left_thres[i]:
            tiles[i][1] = j
            tiles[j][3] = i

    print("Create top neighbours")
    top_thres = np.median(top_matrix, axis=1) / threshold
    top_idx = np.argsort(top_matrix, axis=1)[:, 0]

    for i in range(n_samples):
        j = int(top_idx[i])
        if top_matrix[i, j] < top_thres[i]:
            tiles[i][2] = j
            tiles[j][4] = i

    n_clusters = 1
    tile_to_cluster = defaultdict(int)
    print("Clusters assigning")
    for tile in tqdm(tiles):
        tile_cluster = tile_to_cluster[tile[0]]
        if tile_cluster == 0:
            tile_to_cluster[tile[0]] = n_clusters
            n_clusters += 1
        tile_cluster = tile_to_cluster[tile[0]]
        for idx in tile[1:]:
            if idx is not None:
                if tile_to_cluster[idx] == 0:
                    tile_to_cluster[idx] = tile_cluster
                else:
                    # TODO Think about collisions solution
                    pass

    tiles_dicts = create_mosaic_dict(image_paths, tiles, tile_to_cluster)
    # with open(mosaic_dict_path, 'w') as fout:
    #     json.dump(tiles_dicts, fout, indent=2)

    print("Assemble mosaics")

    mosaics, dicts_to_save = create_mosaics(tiles_dicts)
    remaining_image_paths = []
    success_mosaics = []
    for i, (mosaic, info) in enumerate(zip(mosaics, dicts_to_save)):
        for k in ["i", "j"]:
            indices = sorted(list(set([x[k] for x in info])))
            d = dict(zip(indices, range(len(indices))))
            for x in info:
                x[k] = d[x[k]]
        h = len(set([x["i"] for x in info]))
        w = len(set([x["j"] for x in info]))
        source_w, source_h = SOURCE2SHAPE[source]
        if (w, h) != (source_w, source_h) or len(info) != source_w * source_h:
            remaining_image_paths.extend([x["path"] for x in info])
        else:
            if max([x["i"] for x in info]) + 1 != source_h:
                print(info)
            success_mosaics.append(info)
            cv2.imwrite(
                os.path.join("/data/mosaics", "_".join([osp.basename(x["path"]).split(".")[0] for x in info]) + ".jpg"),
                mosaic,
            )
    return remaining_image_paths, success_mosaics
    # with open(clusters_dict_path, 'w') as fout:
    #     json.dump(dicts_to_save, fout, indent=2)


def main():
    args = parse_args()
    mosaics = []
    for path in glob(args.distances_pattern):
        source = osp.basename(path).split(".")[0]
        distances = np.load(path)
        left_matrix: np.ndarray = distances["left_matrix"]
        top_matrix: np.ndarray = distances["top_matrix"]
        image_paths = distances["paths"].tolist()
        prev_len = len(image_paths)
        threshold = 12
        remaining_image_paths = []
        n_samples = len(image_paths)
        source_mosaics = []
        for i in range(50):
            print(f"STAGE: {i}, len images: {len(image_paths)}, threshold: {threshold}")
            remaining_image_paths, success_mosaics = run(
                i, source, left_matrix, top_matrix, image_paths, threshold=threshold
            )
            source_mosaics.extend(success_mosaics)
            remaining_indices = [image_paths.index(x) for x in remaining_image_paths]
            left_matrix = left_matrix[remaining_indices][:, remaining_indices]
            top_matrix = top_matrix[remaining_indices][:, remaining_indices]
            image_paths = remaining_image_paths
            if prev_len == len(image_paths):
                if threshold == 2.75:
                    break
                threshold = max(threshold * 0.9, 2.75)
            prev_len = len(image_paths)
            if len(image_paths) == 0:
                break
        source_w, source_h = SOURCE2SHAPE[source]
        assert len(remaining_image_paths) + source_w * source_h * len(source_mosaics) == n_samples
        for p in remaining_image_paths:
            copyfile(p, osp.join("/data/mosaics", osp.basename(p)))
        mosaics.extend(source_mosaics)
    import mmcv

    mmcv.dump(mosaics, "/data/mosaics.json")


if __name__ == "__main__":
    main()
