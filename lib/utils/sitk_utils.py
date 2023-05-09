import SimpleITK as sitk
import numpy as np
from multiprocess import Pool
from tqdm import tqdm
from typing import List
from ..custom_types import DatasetDict

def read_spacing(path:str):
    return sitk.ReadImage(path).GetSpacing()

def spacing_from_dataset_json(dataset_dict:DatasetDict,
                              key:str,
                              quantile:float=0.5,
                              n_workers:int=1)->List[float]:
    """
    Calculates the spacing at a given quantile using a dataset dictionary and a
    key.

    Args:
        dataset_dict (DatasetDict): dictionary containing study IDs as values 
            and a dictionary of SITK-readable paths as values.
        key (str): key for the value that will be used to extract the spacing.
        quantile (float, optional): spacing quantile that will be returned. 
            Defaults to 0.5.

    Returns:
        List[float]: spacing at the specified quantile.
    """
    all_spacings = []
    all_paths = [dataset_dict[k][key] for k in dataset_dict
                 if key in dataset_dict[k]]
    with tqdm(all_paths) as pbar:
        pbar.set_description("Inferring target spacing")
        if n_workers > 1:
            pool = Pool(n_workers)
            path_iterable = pool.imap(read_spacing,all_paths)
        else:
            path_iterable = map(read_spacing,all_paths)
        for spacing in path_iterable:
            all_spacings.append(spacing)
            pbar.update()
    all_spacings = np.array(all_spacings)
    output = np.quantile(all_spacings,quantile,axis=0).tolist()
    print("Inferred spacing:",output)
    return output