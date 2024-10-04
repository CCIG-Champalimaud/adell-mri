def filter_dicom_dict_on_presence(data_dict: dict, all_keys: list[str]):
    def check_intersection(a, b):
        return len(set.intersection(set(a), set(b))) == len(set(b))

    for k in data_dict:
        for kk in data_dict[k]:
            data_dict[k][kk] = [
                element
                for element in data_dict[k][kk]
                if check_intersection(element.keys(), all_keys)
            ]
    return data_dict


def filter_dicom_dict_by_size(data_dict: dict, max_size: int):
    print("Filtering by size (max={})".format(max_size))
    output_dict = {}
    removed_series = 0
    for k in data_dict:
        for kk in data_dict[k]:
            if len(data_dict[k][kk]) < max_size:
                if k not in output_dict:
                    output_dict[k] = {}
                output_dict[k][kk] = data_dict[k][kk]
            else:
                removed_series += 1
    print("Removed={}".format(removed_series))
    return output_dict
