import copy, cv2, os
from roboflow import Roboflow
from indices import get_indices
from utils import BoundingBox, convert_xml_to_boxes, count_group_and_type, draw_result, remove_overlap_boxes, sort_upper_to_lower

params = {'width_one_floor' : 1080,
          'num_floors': 29}
groups = ['SPVB', 'NON_SPVB']
drink_types = ['CSD', 'ED', 'JD', 'TEA', 'WATER']

# This model will load the labelling dict to calculate sos; no deep learning
class TestModel():
    def __init__(self):
        self.xml_file = 'data/samples/IMG_0419_08_5_20.xml'
        self.result_dict = {'boxes': convert_xml_to_boxes(self.xml_file)}
        with open('predefined_classes_MT.txt', 'r') as f:
            self.classes = f.read().splitlines()
            
    def analyze_one_image(self, img_path):
        if isinstance(img_path, str):
            img0 = cv2.imread(img_path)
        else:
            img0 = img_path
        self.img = img = copy.deepcopy(img0)
        
        for box in self.result_dict['boxes']:
            if box.label.split('_')[0] == 'SPVB':
                img = draw_result(img, [box], color=(255, 0, 0), put_label=False, put_percent=False)
            else:
                img = draw_result(img, [box], color=(0, 0, 255), put_label=False, put_percent=False)
        cv2.imwrite('data/tmp/test.png', img)
        
        self.sos_dict = self.calculate_sos()
        return self.sos_dict, img
    
    def calculate_sos(self):
        boxes = self.result_dict['boxes']
        floor_total_width = params['width_one_floor'] * params['num_floors']
        num_bottles_per_class = {}
        for cl in self.classes:
            num_bottles_per_class[cl] = 0
        for box in boxes:
            num_bottles_per_class[box.label] += 1
            
        ret_dict = {'total_num_boxes': {}, 'percent': {}}
        for param in ret_dict:
            for group in groups:
                ret_dict[param][group] = {}
                for drink_type in drink_types:
                    ret_dict[param][group][drink_type] = 0.0 if param == 'percent' else 0
                
        for box in boxes:
            key_split = box.label.split('_')
            group = key_split[0] if len(key_split) < 3 else key_split[0] + '_' + key_split[1]
            drink_type = key_split[-1]
            ret_dict['percent'][group][drink_type] += box.w / floor_total_width * 100
            ret_dict['total_num_boxes'][group][drink_type] += 1
            ret_dict['percent'][group][drink_type] = round(ret_dict['percent'][group][drink_type], 1)
            
        ret_dict['percent_skus'] = {}
        skus_spvb = ret_dict['total_num_boxes']['SPVB']
        skus_nonspvb = ret_dict['total_num_boxes']['NON_SPVB']
        num_skus_spvb = sum([skus_spvb[sku] for sku in skus_spvb])
        num_skus_nonspvb = sum([skus_nonspvb[sku] for sku in skus_nonspvb])
        ret_dict['percent_skus']['SPVB'] = round(num_skus_spvb / (num_skus_nonspvb + num_skus_spvb) * 100)
        ret_dict['percent_skus']['NON_SPVB'] = round(num_skus_nonspvb / (num_skus_nonspvb + num_skus_spvb) * 100)
        return ret_dict

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
    # model = Yolov8Model()
    # model.analyze_one_image("data/output/IMG_5841.png")
    model = TestModel()
    a, img = model.analyze_one_image('data/samples/images/IMG_0419_08_5_20.png')    
    
