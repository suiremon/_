import joblib
import pandas as pd


class PredictionModel:
    
    def __init__(self, modelpath):
        
        self.modelpath = modelpath
        self.instance = joblib.load(self.modelpath)
        
    def transform_json(self, json): #изменяет входные данные (оставляет только theory)
        
        self.json = json
        self.df = pd.DataFrame.from_dict(self.json["tasks"])
        self.df["name"] = self.json["name"]
        self.df = self.df[['name', 'type', 'theory', 'tags', 'autoTags']]
        self.df.drop(["autoTags", "name", "type", "tags"], axis = 1, inplace=True)
        return self.df
    
    def predict(self, values): 
        
        self.values = values
        self.values = self.transform_json(self.values)
        self.values["potential_tags"] = ""
        for i in range(len(self.values)):
            
            self.save = (pd.DataFrame(self.instance.predict_proba(self.values.iloc[i])[0], self.instance.classes_, columns = ["acc"])
            .sort_index()
            .sort_values("acc", kind='mergesort')
            .tail(3)
            .reset_index())
            
            self.save = self.save["index"].tolist()
            self.values["potential_tags"].iloc[i] = self.save
        self.values.drop("theory", axis=1, inplace=True)
        self.result = self.values.to_dict('list')

        return self.result #каждому заданию соответствует 3 наиболее вероятных тега
#пока что предсказание основывается только на тексте задания, но с появлением бд можно будет использовать и другие параметры
#предсказание не всегда точное, тк бд небольшая  
