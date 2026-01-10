def count_common_elements(list_one, list_two):
    #  set to not have duplicates
    common_elements = set(list_one).intersection(set(list_two))
    return len(common_elements)