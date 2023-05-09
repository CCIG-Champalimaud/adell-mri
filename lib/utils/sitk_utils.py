import SimpleITK as sitk
import numpy as np
from multiprocess import Pool
from tqdm import tqdm
from typing import List
from typing import Dict,Tuple
from dataclasses import dataclass
from ..custom_types import DatasetDict

@dataclass
class ReadSpacing:
    dataset_dict: DatasetDict
    image_key: str
    
    def __call__(self,key:str)->Tuple[float,float,float]:
        sp = sitk.ReadImage(self.dataset_dict[key][self.image_key]).GetSpacing()
        return key,sp

def spacing_values_from_dataset_json(dataset_dict:DatasetDict,
                                     key:str,
                                     n_workers:int=1)->Dict[str,Tuple[float,float,float]]:
    all_spacings = {}
    read_spacing = ReadSpacing(dataset_dict,key)
    with tqdm(dataset_dict) as pbar:
        pbar.set_description("Inferring target spacing")
        if n_workers > 1:
            pool = Pool(n_workers)
            path_iterable = pool.imap(read_spacing,dataset_dict.keys())
        else:
            path_iterable = map(read_spacing,dataset_dict.keys())
        for key,spacing in path_iterable:
            all_spacings[key] = spacing
            pbar.update()
    return all_spacings

def get_spacing_quantile(spacing_dict:Dict[str,Tuple[float,float,float]],
                         quantile:float=0.5):
    all_spacings = np.array([spacing_dict[k] for k in spacing_dict])
    output = np.quantile(all_spacings,quantile,axis=0).tolist()
    print("Inferred spacing:",output)
    return output

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
    spacing_values = spacing_values_from_dataset_json(
        dataset_dict=dataset_dict,
        key=key,
        n_workers=n_workers)
    output = get_spacing_quantile(spacing_values,quantile)
    return output