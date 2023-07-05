import copy, cv2, os, torch
import numpy as np
from m_utils import convert_xml_to_boxes, convert_preds_to_boxes, count_group_and_type, draw_result

# for yolov7
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load

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

class Yolov7Model():
    def __init__(self, weights='./bestmt0705.pt', img_size=640, conf_thres=0.25, iou_thres=0.45, device='cpu', project='runs/detect'):
        self.device = select_device('cpu')
        self.model = attempt_load(weights, map_location=device)
        self.stride = int(self.model.stride.max())
        self.img_size = img_size
        self.classes = self.model.names
        # self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    
    def predict(self, img, img0):
        with torch.no_grad():
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=None) [0]
        # Rescale boxes from img_size to img0_size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        pred = pred.cpu().detach().numpy()
        return [pr for pr in pred]
    
    def analyze_one_image(self, img_path):
        if isinstance(img_path, str):
            img0 = cv2.imread(img_path) # BGR
        else:
            img0 = img_path
        self.img = img = copy.deepcopy(img0)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        preds = self.predict(img, img0)
        self.result_dict = {'boxes': convert_preds_to_boxes(preds, self.classes)}
        img = self.img
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

        
if __name__ == '__main__':
    model = Yolov7Model()
    results = model.analyze_one_image('data/samples/images/IMG_0419_08_5_20.png')    
    
