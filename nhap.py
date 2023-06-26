from roboflow import Roboflow

class Yolov8Model():
    def __init__(self):
        rf = Roboflow(api_key="YXHyKo6xZrMe72SsXfJK")
        project = rf.workspace().project("spvbtest")
        self.model = project.version(1).model
    
    def predict(self, img_path = "data/output/IMG_5827.png"):
        self.pred_res = self.model.predict(img_path, confidence=40, overlap=30)
        return self.pred_res.json()

    def visualize_res(self, img_path):
        #visualize your prediction
        self.model.predict(img_path, confidence=40, overlap=30).save("prediction.jpg")