import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class XGBoostModel:
    def __init__(self, df, cat_features, num_features, target, mode='dev', test_size=0.2, random_state=42):
        self.df = df
        self.cat_features = [feat.lower() for feat in cat_features]
        self.num_features = [feat.lower() for feat in num_features]
        self.target = target.lower()
        self.mode = mode
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.dtrain = None
        self.dtest = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.process_data()

    def process_data(self):
        self.df = self.process_features(self.df)
        X = self.df[self.cat_features + self.num_features]
        y = self.df[self.target].astype(int)
        if self.mode == 'dev':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            self.align_categories()
            self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, enable_categorical=True)
            self.dtest = xgb.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)
        elif self.mode == 'prod':
            self.X_train, self.y_train = X, y
            self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, enable_categorical=True)

    def align_categories(self):
        for col in self.cat_features:
            self.X_train[col] = self.X_train[col].cat.add_categories(
                [x for x in self.X_test[col].cat.categories if x not in self.X_train[col].cat.categories]
            ).cat.set_categories(self.X_train[col].cat.categories)
            self.X_test[col] = self.X_test[col].cat.set_categories(self.X_train[col].cat.categories)

    def process_features(self, data):
        for column in self.cat_features + self.num_features:
            if column in self.cat_features:
                data[column] = data[column].astype('category')
            else:
                data[column] = pd.to_numeric(data[column], errors='coerce')
        return data

    def train_model(self):
        params = {
            'max_depth': 6,
            'min_child_weight': 1,
            'eta': 0.3,
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'error'  # Using error for binary classification accuracy
        }
        self.model = xgb.train(params, self.dtrain, num_boost_round=100)
        self.evaluate_model()
        if self.mode == 'dev':
            return self.model,self.dtrain,self.X_train,self.dtest,self.X_test
        elif self.mode == 'prod':
            return self.model,self.dtrain,self.X_train,[],[]


    def evaluate_model(self):
        dtest = self.dtest if self.mode == 'dev' else self.dtrain
        predictions = self.model.predict(dtest)
        # Note: For ROC AUC, we do not need to threshold the predictions to 0 or 1
        roc_auc = roc_auc_score(dtest.get_label(), predictions)
        print(f"ROC AUC Score: {roc_auc:.2f}")
