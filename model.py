import copy, cv2, os
from roboflow import Roboflow
from indices import get_indices
from utils import draw_result, remove_overlap_boxes, sort_upper_to_lower

class BoundingBox:
    def __init__(self, res):
        self.x1 = int(res['x'] - res['width']/2)
        self.y1 = int(res['y'] - res['height']/2)
        self.x2 = int(res['x'] + res['width']/2)
        self.y2 = int(res['y'] + res['height']/2)
        self.cen_x = res['x']
        self.cen_y = res['y']
        self.w = res['width']
        self.h = res['height']
        self.prob = res['confidence']
        self.label = res['class']
        self.area = self.w * self.h

class Yolov8Model():
    def __init__(self):
        rf = Roboflow(api_key="YXHyKo6xZrMe72SsXfJK")
        project = rf.workspace().project("spvbtest")
        self.model = project.version(1).model
        #a=self.model.predict("data/output/IMG_5827.png", confidence=40, overlap=30).json()
        #print(a)
    
    def predict(self, img_path):
        pred_res = self.model.predict(img_path, confidence=40, overlap=30)
        pred_res = pred_res.json()
        return pred_res

    def analyze_one_image(self, img_path):
        if isinstance(img_path, str):
            img0 = cv2.imread(img_path)
        else:
            img0 = img_path
        self.img = img = copy.deepcopy(img0)
        # img_key = os.path.basename(img_path).split(".")[0]
        distance_threshold = 1
        
        pred_res = self.predict(img_path)
        #img_shape = pred_res['image']
        boxes = [BoundingBox(res) for res in pred_res['predictions']]
        index_dict = get_indices(boxes)
        boxes = remove_overlap_boxes(
                boxes, exclude_indices=index_dict["bottle"]
        )  # remove overlap shelves
        index_dict = get_indices(boxes)
        boxes = remove_overlap_boxes(
            boxes, exclude_indices=index_dict["shelf"]
        )  # remove overlap bottles
        index_dict = get_indices(boxes)
        index_dict["shelf"] = sort_upper_to_lower(boxes, index_dict["shelf"])
        index_dict["bottle"] = sort_upper_to_lower(boxes, index_dict["bottle"])
        img = draw_result(
                img,
                [boxes[i] for i in index_dict["bottle"]],
                color=(192, 192, 192),
                put_label=False,
                put_percent=False,
            )
        img = draw_result(
            img,
            [boxes[i] for i in index_dict["shelf"]],
            color=(192, 192, 192),
            put_label=False,
            put_percent=False,
        )
        
        # list_bottles only counts bottles on shelves; while bottle_indices includes all bottles
        list_bottles = self.assign_shelves(
            boxes,
            index_dict["shelf"],
            index_dict["bottle"],
            distance_threshold=distance_threshold,
        )
        for shelf_idx in range(len(index_dict["shelf"])):
            list_bottles_one_shelf = list_bottles[f"F{shelf_idx + 1}"]
            list_bottles_one_shelf["Main layer"].sort(key=lambda box: box.x1)
            list_bottles_one_shelf_all = (list_bottles_one_shelf["Main layer"] + list_bottles_one_shelf["Sublayer"])

            # Drawing
            shelf = boxes[index_dict["shelf"][shelf_idx]]
            img = draw_result(img, list_bottles_one_shelf_all, color=(0, 255, 0), put_label=True, put_percent=False)
            img = draw_result(img, [shelf], color=(255, 0, 0), put_label=True, put_percent=False)    
            
        self.res_img = img
        cv2.imwrite('res.png', img)
        # Calculate Share-Of-Shelf (sos)
        self.sos_dict = self.calculate_sos(list_bottles, [boxes[idx] for idx in index_dict['shelf']])
        return self.sos_dict, self.res_img
        
        
    # which bottles belong to which shelves
    def assign_shelves(self, boxes, shelf_indices, bottle_indices, distance_threshold):
        ret = {}
        flag = [True for _ in range(len(bottle_indices))]
        for si, shelf_idx in enumerate(shelf_indices):
            # Main layer lies directly on a shelf; sublayer is only to check if there is Non-SPVB product
            ret[f"F{si + 1}"] = {"Main layer": [], "Sublayer": []}
            for bi, bottle_idx in enumerate(bottle_indices):
                # If bottom of bottle is on a shelf and top of bottle is above the shelf
                if boxes[shelf_idx].y2 >= boxes[bottle_idx].y2 and flag[bi]:
                    if (
                        boxes[shelf_idx].y1 - boxes[bottle_idx].y2
                        < boxes[bottle_idx].h / distance_threshold
                    ):
                        ret[f"F{si + 1}"]["Main layer"].append(boxes[bottle_idx])
                        flag[bi] = False
                    else:
                        ret[f"F{si + 1}"]["Sublayer"].append(boxes[bottle_idx])
                        flag[bi] = False
        return ret
    
    def calculate_sos(self, list_bottles, list_shelves):
        assert len(list_bottles) == len(list_shelves), 'Bottles assigned to different number of floors'
        sos_dict = {'sos': 0.0, 'SPVB': 0, 'NonSPVB': 0}
        floor_width_accu = 0.0
        bottle_width_accu = 0.0
        for i, floor in enumerate(list_bottles):
            floor_width_accu += list_shelves[i].w
            for j, bottle in enumerate(list_bottles[floor]['Main layer']):
                bottle_width_accu += bottle.w
                if bottle.label == 'SPVB':
                    sos_dict['SPVB'] += 1
                else:
                    sos_dict['NonSPVB'] += 1                
        sos_dict['sos'] = round(bottle_width_accu / floor_width_accu, 3)
        return sos_dict
        
if __name__ == '__main__':
    model = Yolov8Model()
    model.analyze_one_image("data/output/IMG_5841.png")