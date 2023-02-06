import numpy as np
import torch
import monai
from copy import deepcopy
from collections import abc

from typing import Sequence,Callable,Union,Tuple,Dict

DICOMDatasetType = Sequence[Dict[str,Sequence[Dict[str,str]]]]

def filter_orientations(
    dicom_dictionary:DICOMDatasetType,
    keep_bad:bool=True)->DICOMDatasetType:
    """Filters out DCM files with bad orientations (i.e. those which have
    a "orientation" key whose last three elements are close to [0,0,-1]).

    Args:
        dicom_dictionary (DICOMDatasetType): a DICOM dataset.
        keep_bad (bool): keeps the bad annotations and eliminates only DICOM
            files with no orientation tags.

    Returns:
        DICOMDatasetType: a filtered DICOM dataset.
    """
    print("Filtering out bad orientations")
    new_dicom_dictionary = {}
    for k in dicom_dictionary:
        new_dicom_dictionary[k] = {}
        for kk in dicom_dictionary[k]:
            new_dicom_dictionary[k][kk] = []
            for dcm_dict in dicom_dictionary[k][kk]:
                if dcm_dict["orientation"] is not None:
                    is_bad = np.all(
                        np.isclose(
                            dcm_dict["orientation"][-3:],[0,0,-1]))
                    if is_bad == True and keep_bad == False:
                        pass
                    else:
                        new_dicom_dictionary[k][kk].append(dcm_dict)

    new_dict = {}
    for k in new_dicom_dictionary:
        new_sub_dict = {}
        for kk in new_dicom_dictionary[k]:
            if len(new_dicom_dictionary[k][kk]) > 0:
                new_sub_dict[kk] = new_dicom_dictionary[k][kk]
        if len(new_sub_dict) > 0:
            new_dict[k] = new_sub_dict

    return new_dict

class DICOMDataset(torch.utils.data.Dataset):
    """
    Very similar to the standard MONAI Dataset but supports nested indexing.
    The dataset is expected to be a list of dictionaries, where each key 
    corresponds to a sequence_uid and each value corresponds to a list of 
    dictionaries with labelled paths to one DICOM file. I.e.:
    
    [{sequence_uid_1:[{"image":"image-001.dcm"},{"image":"image-002.dcm"}],
      sequence_uid_2:[{"image":"image-003.dcm"},{"image":"image-004.dcm"}]},
     {sequence_uid_1:[{"image":"image-005.dcm"},{"image":"image-006.dcm"}],
      sequence_uid_2:[{"image":"image-007.dcm"},{"image":"image-008.dcm"}]}]
    """
    
    def __init__(self, 
                 dicom_dataset: DICOMDatasetType, 
                 transform: Callable=None) -> None:
        """
        Args:
            dicom_dataset (DICOMDatasetType): input data to load and transform 
                to generate dataset for model as specified above.
            transform (Callable): a callable data transform on input data.
                Defaults to None (no transform).
        """
        self.dicom_dataset = dicom_dataset
        self.transform = transform
        
        self.correspondence = self.get_correspondence(self.dicom_dataset)
    
    @staticmethod
    def get_correspondence(dicom_dataset):
        """
        Constructs a correspondence between numeric indices and specific 
        studies and series.
        """
        correspondence = []
        for index,element in enumerate(dicom_dataset):
            for k in element:
                for dcm_index,_ in enumerate(element[k]):
                    correspondence.append([index,k,dcm_index])
        return correspondence

    def __len__(self) -> int:
        return len(self.correspondence)

    def _transform(self, index: Union[int,Tuple[int,str,int]]):
        """
        Fetch single data item from `self.dicom_dataset`.
        """
        if isinstance(index,int):
            real_index = self.correspondence[index]
        else:
            real_index = index
        data_i = self.dicom_dataset[real_index[0]][real_index[1]][real_index[2]]
        if isinstance(data_i,str):
            print(real_index,data_i)
        if self.transform is not None:
            return monai.transforms.apply_transform(self.transform, data_i)
        else:
            return data_i

    def __getitem__(self, index: Union[int, slice, Sequence]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item 
        otherwise.
        """
        if isinstance(index, tuple):
            return self._transform(index)
        elif isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return torch.utils.data.Subset(dataset=self, indices=indices)
        elif isinstance(index, abc.Sequence):
            # dataset[[1, 3, 4]]
            return torch.utils.data.Subset(dataset=self, indices=index)
        return self._transform(index)

class SliceSampler(torch.utils.data.Sampler):
    """
    Sampler that yields n_iterations slices for each study in dicom_dataset.
    """
    def __init__(self,
                 dicom_dataset:DICOMDatasetType,
                 n_iterations:int=1,
                 shuffle:bool=True,
                 seed:int=42):
        """
        Args:
            dicom_dataset (DICOMDatasetType): input data with the same format 
                as specified in DICOMDataset.
            n_iterations (int, optional): number of times a study is accessed 
                by epoch. Defaults to 1.
            shuffle (bool, optional): whether the indices should be shuffled 
                before each epoch. Defaults to True.
            seed (int, optional): random seed. Defaults to 42.
        """
        self.dicom_dataset = dicom_dataset
        self.n_iterations = n_iterations
        self.shuffle = shuffle
        self.seed = seed
        
        self.keys_to_indices()
        self.rng = np.random.default_rng(self.seed)
        
    def keys_to_indices(self):
        """
        Constructs a correspondence between numeric indices and specific 
        studies and series.
        """
        self.correspondence = []
        self.N = 0
        self.i = 0
        for element in self.dicom_dataset:
            new_element = {}
            for k in element:
                new_element[k] = []
                for _ in element[k]:
                    new_element[k].append(self.i)
                    self.i += 1
                self.correspondence.append(new_element)
                self.N += 1
        
    def __iter__(self)->int:
        """Returns indices for DICOMDataset such that only one sample for each
        study is returned. Additionally, if `shuffle==True`, the elements are
        shuffled before iterating over the DICOM dataset list. The DICOM 
        dataset can also be iterated n_iterations times (this can be helpful 
        in defining the data used in different epochs, i.e. "each epoch uses
        samples a slice from each study n_iterations times".

        Yields:
            int: an index used for the __getitem__ method in DICOMDataset.
        """
        corr_idx = []
        for _ in range(self.n_iterations):
            corr_idx.extend([i for i in range(self.N)])
            
        if self.shuffle == True:
            self.rng.shuffle(corr_idx)
        
        for idx in corr_idx:
            element = self.correspondence[idx]
            idx = int(self.rng.choice(
                element[self.rng.choice(list(element.keys()))]))
            yield idx

    def __len__(self)->int:
        """Length of the dataset (number of studies). The number of individual
        dcm files can also be accessed with `self.i`.

        Returns:
            int: number of studies.
        """
        return self.N * self.n_iterations
