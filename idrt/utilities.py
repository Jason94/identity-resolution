from typing import List, Tuple

from data import Field


def split_field_dict(
    fields: List[Field], data: List[dict]
) -> Tuple[List[dict], List[dict]]:
    data1 = []
    data2 = []

    for d in data:
        d1 = {}
        d2 = {}

        for f in fields:
            d1[f.field] = d[f.field + "1"]
            d2[f.field] = d[f.field + "2"]

        data1.append(d1)
        data2.append(d2)

    return data1, data2


def transpose_dict_of_lists(dict_of_lists):
    keys = dict_of_lists.keys()
    length_of_lists = len(next(iter(dict_of_lists.values())))

    list_of_dicts = []
    for i in range(length_of_lists):
        new_dict = {}
        for key in keys:
            new_dict[key] = dict_of_lists[key][i]
        list_of_dicts.append(new_dict)

    return list_of_dicts
