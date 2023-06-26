def get_shelf_indices(boxes):
    return [i for i, box in enumerate(boxes) if box.label=="Splitline"]

def get_bottle_indices(boxes):
    return [i for i, box in enumerate(boxes) if box.label=="TH" or box.label=="SPVB"]

def get_indices(boxes):
    ret_dict = {}
    shelf_indices = get_shelf_indices(boxes)
    bottle_indices = get_bottle_indices(boxes)
    ret_dict["shelf"] = shelf_indices
    ret_dict["bottle"] = bottle_indices
    return ret_dict