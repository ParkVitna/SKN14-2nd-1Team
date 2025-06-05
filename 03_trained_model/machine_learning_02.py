import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import joblib

def data_load():
    # data_df = pd.read_csv('data/internet_service_churn_after_eda.csv')
    data_df = pd.read_csv('../data/train.csv')

    # EDA에서 진행된 부분 주석 처리
    # 칼럼명 오타 수정
    # data_df.rename(columns={'reamining_contract': 'remaining_contract'}, inplace=True)
    # id 제거
    data_df.drop(columns=["id"], inplace=True)
    # 결측치 0으로 채우기
    # data_df = data_df.fillna(0)

    # subscription_status_label (None, TV only, Movie only, Both) 제거
    # data_df.drop(columns=["subscription_status_label"], inplace=True)

    # remaining_contract 칼럼의 상관관계가 다른 칼럼에 비해 매우 높아 제거 후 진행
    data_df.drop(columns=["remaining_contract"], inplace=True)

    # is_tv_subscriber, is_movie_package_subscriber -> subscription_status 통합 칼럼으로 제거
    data_df.drop(columns=["is_tv_subscriber", "is_movie_package_subscriber"], inplace=True)

    return data_df

def data_split(data_df, scaler_nm='None'):
    X = data_df.drop(columns=["churn"])
    y = data_df["churn"]

    # 범주형 및 수치형 컬럼 분리
    cat_feature_names = ["subscription_status"]
    num_feature_names = [col for col in X.columns if col not in cat_feature_names]

    X_num = X[num_feature_names]
    X_cat = X[cat_feature_names].astype(str)  # 반드시 문자열로 변환

    # XGBoost를 위한 범주형 데이터 LabelEncoding 사용 수치형으로 변환
    # 범주형 라벨 인코딩
    X_cat_encoded = pd.DataFrame()
    for col in cat_feature_names:
        le = LabelEncoder()
        X_cat_encoded[col] = le.fit_transform(X_cat[col])

    # 최종 데이터 통합
    X_processed = pd.concat([X_num, X_cat_encoded], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

    # 스케일링
    if scaler_nm == 'standard':
        scaler = StandardScaler()
    elif scaler_nm == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_nm == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaler_nm == 'quantile':
        scaler = QuantileTransformer()
    elif scaler_nm == 'robust':
        scaler = RobustScaler()
    elif scaler_nm is None or scaler_nm.lower() == 'none':
        scaler = None
    else:
        raise ValueError(f"Unknown scaler name: {scaler_nm}")

    if scaler is not None:
        X_train_num_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[num_feature_names]),
            columns=num_feature_names,
            index=X_train.index
        )
        X_test_num_scaled = pd.DataFrame(
            scaler.transform(X_test[num_feature_names]),
            columns=num_feature_names,
            index=X_test.index
        )
    else:
        X_train_num_scaled = X_train[num_feature_names].copy()
        X_test_num_scaled = X_test[num_feature_names].copy()

    # 범주형 데이터는 그대로 사용
    X_train_final = pd.concat([X_train_num_scaled, X_train[cat_feature_names]], axis=1)
    X_test_final = pd.concat([X_test_num_scaled, X_test[cat_feature_names]], axis=1)

    return X_train_final, X_test_final, y_train, y_test



class ChurnModelTrainer:
    def __init__(self, data):
        self.data = data
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(),
            'robust': RobustScaler(),
            'none': None
        }
        self.cat_features = ["subscription_status"]
        self.results = []


    def preprocess(self, scaler_name, for_catboost=True):
        X = self.data.drop(columns=["churn"])
        y = self.data["churn"]
        num_features = [col for col in X.columns if col not in self.cat_features]

        # 범주형 처리
        X_cat = X[self.cat_features].copy()
        if for_catboost:
            X_cat = X_cat.astype(str)
        else:
            for col in self.cat_features:
                le = LabelEncoder()
                X_cat[col] = le.fit_transform(X_cat[col].astype(str))

        # 스케일링

        if scaler_name == 'none':
            X_num_scaled = pd.DataFrame(X[num_features], columns=num_features)
        else:
            scaler = self.scalers[scaler_name]
            X_num_scaled = pd.DataFrame(scaler.fit_transform(X[num_features]), columns=num_features)

        X_processed = pd.concat([X_num_scaled, X_cat], axis=1)
        return train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)


    def train_catboost(self, X_train, X_test, y_train, y_test, scaler_name):
        model = CatBoostClassifier(verbose=0, random_state=42)
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.1],
            'iterations': [200, 500, 800]
        }
        train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
        test_pool = Pool(X_test, y_test, cat_features=self.cat_features)

        grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
        grid.fit(X_train, y_train, cat_features=self.cat_features)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        self.results.append({
            'model': 'CatBoost',
            'scaler': scaler_name,
            'params': grid.best_params_,
            'f1': report['1']['f1-score'],
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'estimator': best_model
        })


    def train_xgboost(self, X_train, X_test, y_train, y_test, scaler_name):
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        param_grid = {
            # 'max_depth': [4, 6, 8],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.01, 0.03, 0.1],
            # 'n_estimators': [200, 500, 800]
            'n_estimators': [800, 1000, 1200]
        }

        grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        self.results.append({
            'model': 'XGBoost',
            'scaler': scaler_name,
            'params': grid.best_params_,
            'f1': report['1']['f1-score'],
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'estimator': best_model
        })


    def run_all(self):
        for scaler_name in self.scalers:
            # CatBoost
            X_train, X_test, y_train, y_test = self.preprocess(scaler_name, for_catboost=True)
            self.train_catboost(X_train, X_test, y_train, y_test, scaler_name)

            # XGBoost
            X_train, X_test, y_train, y_test = self.preprocess(scaler_name, for_catboost=False)
            self.train_xgboost(X_train, X_test, y_train, y_test, scaler_name)


    def save_best_model(self, save_path='../best_model.pkl'):
        best = max(self.results, key=lambda x: (x['f1'], x['accuracy'], x['roc_auc']))
        joblib.dump(best['estimator'], save_path)
        print(f"\n== Best Model Saved: {best['model']} + {best['scaler']} scaler ==")
        print(f"F1-score: {best['f1']:.4f}, Accuracy: {best['accuracy']:.4f}, ROC AUC: {best['roc_auc']:.4f}")
        print(f"\n== Best Parameters: {best['params']} ==")
        return best

if __name__ == '__main__':
    # 실행 부분
    data = data_load()
    trainer = ChurnModelTrainer(data)
    trainer.run_all()
    best_model_info = trainer.save_best_model()

    # 성능 비교 테이블
    results_df = pd.DataFrame(trainer.results)
    results_df_sorted = results_df.sort_values(by=['f1', 'accuracy', 'roc_auc'], ascending=False).reset_index(drop=True)
    print('\n')
    # 모든 열 출력
    pd.set_option('display.max_columns', None)
    # 컬럼 내 문자열 길이 제한 해제
    pd.set_option('display.max_colwidth', None)
    # 줄 바꿈 없이 한 줄에 출력
    pd.set_option('display.expand_frame_repr', False)
    print(results_df_sorted[['model', 'scaler', 'f1', 'accuracy', 'roc_auc', 'params']])


# == Best Model Saved: CatBoost + quantile scaler ==
# F1-score: 0.8414, Accuracy: 0.8285, ROC AUC: 0.9066
#
# == Best Parameters: {'depth': 6, 'iterations': 800, 'learning_rate': 0.1} ==
#
#
#        model    scaler        f1  accuracy   roc_auc                                                          params
# 0   CatBoost  quantile  0.841423  0.828520  0.906630           {'depth': 6, 'iterations': 800, 'learning_rate': 0.1}
# 1    XGBoost  standard  0.840587  0.828001  0.905126  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1200}
# 2    XGBoost    minmax  0.840587  0.828001  0.905126  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1200}
# 3    XGBoost    maxabs  0.840587  0.828001  0.905126  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1200}
# 4    XGBoost    robust  0.840587  0.828001  0.905126  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1200}
# 5    XGBoost      none  0.840587  0.828001  0.905126  {'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1200}
# 6    XGBoost  quantile  0.840058  0.827222  0.905497    {'learning_rate': 0.03, 'max_depth': 8, 'n_estimators': 800}
# 7   CatBoost  standard  0.839706  0.826704  0.906283           {'depth': 8, 'iterations': 500, 'learning_rate': 0.1}
# 8   CatBoost    minmax  0.839706  0.826704  0.906283           {'depth': 8, 'iterations': 500, 'learning_rate': 0.1}
# 9   CatBoost    maxabs  0.839706  0.826704  0.906283           {'depth': 8, 'iterations': 500, 'learning_rate': 0.1}
# 10  CatBoost      none  0.839706  0.826704  0.906283           {'depth': 8, 'iterations': 500, 'learning_rate': 0.1}
# 11  CatBoost    robust  0.839706  0.826704  0.906282           {'depth': 8, 'iterations': 500, 'learning_rate': 0.1}
