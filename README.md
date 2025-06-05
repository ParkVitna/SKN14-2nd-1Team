# SKN14-2nd-1Team

# 👣팀 소개

## 팀명 - 이탈하지말아조

<table>
  <tr>
    <th>김재아</th>
    <th>박빛나</th>
    <th>이승철</th>
    <th>하종수</th>
    <th>한성규</th>
  </tr>
  <tr>
    <td><img src="https://i.namu.wiki/i/ZILsKyXWcHM8uj_n8GzghV5TodK2rlwkZYbBcIQQ89NGKPdKlrjB6o-O0kEPyDNhEHpekJ-TeujwOSbrzp0llA.svg" width="150"></td>
    <td><img src="https://cdn.seoulcity.co.kr/news/photo/202008/403047_203101_2112.png" width="150"></td>
    <td><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGWZg8FqLLI0fK-6dbrQJCpnLTExJzdH6o1Q&s" width="150"></td>
    <td><img src="https://i.namu.wiki/i/kWB5ni5OxQom0vmSxpmCsw-R76z2LLrObeR_2PO7Jc49li30-5QWBYXLPjklh_WbcYYnldbIjDCf5awkqxip7g.svg" width="150"></td>
    <td><img src="https://i.namu.wiki/i/OIXVGcqwZqkVv546uWnq2Sbtm0mU8xGugk1XDiM8k2jOIPv0PjCXto8HTS_vb-9403_lRNdYLDtdGKjSPNek9g.svg" width="150"></td>
  </tr>
  <tr>
    <td><a href="https://github.com/sanjum-kim"><img src="https://img.shields.io/badge/GitHub-Sanjum_Kim-green?logo=github"></a></td>
    <td><a href="https://github.com/ParkVitna"><img src="https://img.shields.io/badge/GitHub-Vitna-pink?logo=github"></a></td>
    <td><a href="https://github.com/ezcome-ezgo"><img src="https://img.shields.io/badge/GitHub-Ezcome_Ezgo-blue?logo=github"></a></td>
    <td><a href="https://github.com/ha1153"><img src="https://img.shields.io/badge/GitHub-Ha1153-lightblue?logo=github"></a></td>
    <td><a href="github.com/Seonggyu-Han"><img src="https://img.shields.io/badge/GitHub-Seonggyu_Han-red?logo=github"></a></td>
  </tr>
</table>

# 📄 데이터셋 개요
🛰️ 인터넷 구독 서비스 이용객 이탈률 예측 데이터 셋

Internet Service Provider Customer Churn - <br>

"https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn"

선택이유 - 1인 가구 증가, 통신사의 데이터 서비스 및 신뢰감 변화등으로 인해 인터넷 구독에 관한 소비자의 선택 폭이 다양해지고 있다. 이러한 상황에서
회사는, 기존 고객을 붙잡고 새로운 고객을 유치하며, 이탈 고객을 다시 돌아오게 해야 한다. 이러한 방안을 제안하기 위해, 고객 이탈률을 예측하여 대안 방안을 제시 할 예정이다.


<br><center>인터넷(유선)사용량 변화</center>

![news](images/111.png)

<br><center> 서비스 유지 보수 문제 </center>

![news](images/112.png)


## 🧮 데이터프레임 구조
총 행 수 (entries): 72,274

총 열 수 (columns): 11

메모리 사용량: 약 6.1 MB

| 열 이름                          | Not null 값 | 데이터 타입  |
| ----------------------------- |------------| ------- |
| `id`                          | 72,274     | int64   |
| `is_tv_subscriber`            | 72,274     | int64   |
| `is_movie_package_subscriber` | 72,274     | int64   |
| `subscription_age`            | 72,274     | float64 |
| `bill_avg`                    | 72,274     | int64   |
| `reamining_contract`          | 50,702     | float64 |
| `service_failure_count`       | 72,274     | int64   |
| `download_avg`                | 71,893     | float64 |
| `upload_avg`                  | 71,893     | float64 |
| `download_over_limit`         | 72,274     | int64   |
| `churn`                       | 72,274     | int64   |


📏 데이터프레임 크기
(행, 열) = (72,274, 11)

# 🗂️ 컬럼 설명

| 컬럼명                           | 설명                                                      |
| ----------------------------- | ------------------------------------------------------- |
| `id`                          | 고유 구독자 ID                                               |
| `is_tv_subscriber`            | TV 구독 여부                                                |
| `is_movie_package_subscriber` | 영화 패키지 구독 여부                                            |
| `subscription_age`            | 구독 기간                                                   |
| `bill_avg`                    | 지난 3개월 평균 청구 금액                                         |
| `reamining_contract`          | 계약의 남은 연수<br>(Null일 경우 계약 없음, 계약 종료 전 해지 시 위약금 발생)      |
| `service_failure_count`       | 지난 3개월간 서비스 장애로 인한 콜센터 고객 통화 횟수                         |
| `download_avg`                | 지난 3개월 평균 다운로드 사용량 (GB)                                 |
| `upload_avg`                  | 지난 3개월 평균 업로드 사용량 (GB)<br>※ 다운로드 및 업로드 평균 사용량은 3GB 제한 내 |
| `download_over_limit`         | 지난 9개월 동안 다운로드 제한 초과 횟수<br>※ 초과 시 추가 요금 발생              |
| `churn`                       | 서비스 취소 여부                                               |

## 데이터 전처리 및 Feature Engineering

### 결측값 전처리

 - Col[reamining contract]항목 -> [remaining contract]로 수정(오타)

 - [remaining contract] 결측치 = 21572

 - download_avg, upload_avg 결측치 381

📌 remaining_contract 변수에 대한 해석 및 처리 기준

remaining_contract는 **계약의 남은 기간(연 단위)**을 의미하며,
NaN은 "계약 기간이 없는 경우"로 이해됨.

이탈자의 경우 remaining_contract를 0으로 간주해도 무방.

하지만 이탈하지 않았지만 값이 NaN인 경우, 이는 실제 결측치(missing value)로 판단하는 것이 더 타당함.

🔍 0 값의 의미 분석
연 단위로 환산 시, 1일 ≈ 0.00274

따라서 remaining_contract = 0.01은 약 3.6일, 즉 4일 정도를 의미함.

이 데이터셋에서는 4일 이상 남은 경우만 0.01 이상의 값으로 표기,
4일 미만은 모두 0으로 표기된 것으로 보임.

✅ 결론 및 처리 기준

remaining_contract = 0인 경우는 계약 만료 임박 또는 종료 상태로 판단 가능.

실제로 **해당 값들의 churn은 거의 1(이탈)**로 나타나므로,
0도 NaN과 마찬가지로 계약 종료 상태로 간주하는 것이 타당함.

### ⚙️Feature Engineering
- 현재 TV 구독, 영화 구독에 대해 각각 이탈 or 유지만 분석함,
- 하지만, TV 구독과 영화 구독을 둘 다 하지 않는 경우, 둘 다 하는 경우, 하나만 하는 경우로 데이터들이 기록되어있기 때문에 새롭게 subscription_status 라는 column을 만들어서
0. : 구독 없음
1. : TV 구독
2. : 영화 구독
3. : 둘 다 구독


과 같은 형태로 새로 만드는 방법으로 새롭게 상관성을 찾고 시각화하여  확인하는게 좋지 않을까 라는 의문에서 새로운 컬럼 생성


# 머신 러닝을 위한 모델 및 스케일러 선택

## 📊 사용 모델 설명 및 적합성

| 모델                                  | 설명                                                                | 데이터셋과의 적합성                                                               |
| ----------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **XGBClassifier**                   | Gradient Boosting 기반의 고성능 트리 모델. 정교한 에러 보정과 regularization 기능 포함. | 다수의 이진 범주형 및 수치형 데이터가 혼합되어 있고, 결측값 처리와 학습 속도가 뛰어나 실무에서 churn 예측에 자주 사용됨. |
| **CatBoostClassifier**              | 범주형 변수 자동 처리에 최적화된 Gradient Boosting 모델.                          | 이 데이터셋은 이진형 변수들이 많아 CatBoost의 자동 인코딩 기능이 매우 유리함. 튜닝 없이도 성능이 우수.          |
| **SVC (Support Vector Classifier)** | 데이터 간 경계를 최대한 넓히는 초평면을 찾는 모델. 고차원에서 효과적.                          | 피처 수가 과도하지 않으며, 이진 분류 문제로 적합. 하지만 스케일러에 민감하여 적절한 전처리가 매우 중요함.            |
| **RandomForestClassifier**          | 여러 결정 트리를 앙상블해 예측 정확도를 높이는 모델. 과적합에 강하고 해석도 용이.                   | 변수 간 상호작용을 잘 잡으며, 결측치가 있는 상태에서도 비교적 견고하게 작동함.                            |
| **LogisticRegression**              | 고전적인 선형 모델. 해석력이 뛰어나고 구현이 간단.                                     | 기준선 모델(Baseline)로 매우 적절. 스케일 조정에 민감하지만 간단한 문제의 경우에도 유의미한 결과를 낼 수 있음.     |

## 🧪 사용 스케일러 설명 및 적합성

| 스케일러                    | 설명                                               | 데이터셋과의 적합성                                                                                                |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **None**                | 전처리 없이 원본 그대로 사용.                                | 기준선 비교용. 어떤 모델이 스케일에 민감한지 파악할 수 있음.                                                                       |
| **StandardScaler**      | 평균 0, 표준편차 1로 정규화. 정규 분포 가정이 있는 모델에 적합.          | `subscription_age`, `bill_avg`, `download_avg` 등의 값들이 정규 분포에 가까울 경우 효과적. SVC, LogisticRegression 등과 잘 작동. |
| **MinMaxScaler**        | 모든 값을 \[0, 1] 범위로 조정. 이상치에 민감.                   | 대부분 피처가 정해진 범위를 가지므로 잘 작동할 수 있으나, 이상치(outlier)에는 취약.                                                      |
| **RobustScaler**        | 중앙값과 IQR을 기준으로 스케일링. 이상치에 강건.                    | `bill_avg`나 `service_failure_count`처럼 이상값이 있을 수 있는 변수에 적합.                                                |
| **MaxAbsScaler**        | 최대 절대값을 기준으로 -1\~1 범위로 스케일. 희소 행렬에 적합.           | 이진 변수 또는 0에 가까운 데이터가 많을 경우 유리하지만, 이 데이터셋에는 제한적인 장점.                                                       |
| **QuantileTransformer** | 데이터를 정규 분포 또는 균등 분포로 변환. 이상치를 제거하면서 모델 성능 향상 가능. | 비정규 분포 데이터에 강력하고, 특히 Logistic/SVC와 같은 스케일 민감 모델에 효과적.                                                     |

## ✅ 요약 추천

스케일러가 중요한 모델: SVC, LogisticRegression

스케일러 없이도 잘 작동하는 모델: XGBClassifier, CatBoost, RandomForest

범주형/이진형 특성 고려: CatBoost 매우 유리

해석 가능성과 속도 고려: LogisticRegression, RandomForest 적합

고성능과 일반화: XGBClassifier, CatBoost 우수

# 모델의 성능지표 확인

## 사전조건

1. 결측치는 0으로 대체하여 진행한다.
2. 하이퍼 파라미터는 random_state = 42만 사용하여 모든 모델을 고정하고 비교한다.
2. ['id'] 컬럼은 학습에 필요 없으니 제거한다.
3. ['remaining_contract'] 컬럼은 과적합의 우려가 있어 비교하기 위해 컬럼을 유지한 것과 삭제한 것을 비교한다.
4. Feature Engineering으로 인한 Label 값과, 중복 학습을 일으킬 수 있는 ['is_tv_subscriber', 'subscription_status_label', 'is_movie_package_subscriber'] 컬럼 또한 제거하여 학습한다.
5. 적합한 모델을 확인하기 위해 사전에 5가지 모델의 6가지 스케일의 값을 비교하여 최적의 스케일링인 Quantiltransformer와 스케일링을 하지 않은 값을 비교한다.

## 📊 이탈 예측 모델 성능 비교 (하이퍼 파라미터 X)

✅ remaining_contract 유지 + Quantiltransformer

| Model               | Precision    | Recall       | F1-score     | Accuracy     | ROC AUC      |
| ------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| CatBoost            | **0.946964** | 0.941792     | 0.944371     | 0.938516     | **0.978397** |
| XGBoost             | 0.946423     | **0.945537** | **0.945980** | **0.940159** | 0.978164     |
| Random Forest       | 0.946780     | 0.938358     | 0.942550     | 0.936614     | 0.976256     |
| Logistic Regression | 0.849964     | 0.915886     | 0.881695     | 0.863801     | 0.931560     |
| SVM                 | 0.851464     | 0.875780     | 0.863451     | 0.846506     | 0.922671     |

✅ remaining_contract 유지 + 스케일링 없음

| Model               | Precision    | Recall       | F1-score     | Accuracy     | ROC AUC      |
| ------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| CatBoost            | **0.946964** | 0.941792     | 0.944371     | 0.938516     | **0.978397** |
| XGBoost             | 0.946423     | **0.945537** | **0.945980** | **0.940159** | 0.978164     |
| Random Forest       | 0.946780     | 0.938358     | 0.942550     | 0.936614     | 0.976256     |
| Logistic Regression | 0.849964     | 0.915886     | 0.881695     | 0.863801     | 0.931560     |
| SVM                 | 0.851464     | 0.875780     | 0.863451     | 0.846506     | 0.922671     |

✅ remaining_contract 제거 + Quantiltransformer

| Model               | Precision    | Recall       | F1-score     | Accuracy     | ROC AUC      |
| ------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| CatBoost            | 0.861735     | **0.819913** | 0.840304     | 0.827309     | **0.905883** |
| XGBoost             | **0.864379** | 0.817572     | **0.840324** | **0.827828** | 0.905477     |
| Random Forest       | 0.850670     | 0.802747     | 0.826014     | 0.812608     | 0.890576     |
| SVM                 | 0.801214     | 0.720818     | 0.758893     | 0.746195     | 0.822489     |
| Logistic Regression | 0.728006     | 0.789014     | 0.757283     | 0.719734     | 0.803120     |

✅ remaining_contract 제거 + 스케일링 없음

| Model               | Precision    | Recall       | F1-score     | Accuracy     | ROC AUC      |
| ------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| CatBoost            | 0.861735     | **0.819913** | 0.840304     | 0.827309     | **0.905883** |
| XGBoost             | **0.864379** | 0.817572     | **0.840324** | **0.827828** | 0.905477     |
| Random Forest       | 0.850670     | 0.802747     | 0.826014     | 0.812608     | 0.890576     |
| SVM                 | 0.801214     | 0.720818     | 0.758893     | 0.746195     | 0.822489     |
| Logistic Regression | 0.728006     | 0.789014     | 0.757283     | 0.719734     | 0.803120     |

✔️ 최적의 머신러닝 학습 모델과 스케일러 결론

- ['remaining_contract'] 컬럼을 삭제하지 않았을 때, 높은 점수를 얻을 수 있지만 과적합 우려로 인해 삭제하고 학습하는 것이 좋다.
- 컬럼을 삭제했을 때의 점수는 XGBoost 모델이 1등, 그 다음을 CatBoost 모델이 잇고 있다.
- 현재의 데이터 셋에 XGBoost와 CatBoost, 두 모델을 사용하고 하이퍼 파라미터를 수정하여 더 높은 값을 찾아내야 한다.

## XGBoost & CatBoost 성능 향상을 위한 하이퍼 파라미터값 조정

### 하이퍼 파라미터 조건

| 하이퍼파라미터         | 선택한 값들          | 의미 및 역할                                | 모델 성능에 미치는 영향                                                                              |
| --------------- | --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------ |
| `depth`         | 4, 6, 8         | 각 트리의 최대 깊이.<br>값이 클수록 더 복잡한 패턴 학습 가능. | - **작을수록** 과적합 위험은 적지만 학습한 정보가 제한됨<br>- **클수록** 더 많은 패턴을 포착하지만 **과적합** 가능성 증가|
| `learning_rate` | 0.01, 0.03, 0.1 | 각 트리의 기여 정도. 학습 속도 조절.                 | - **작을수록** 안정적인 학습 가능, 하지만 많은 트리 필요<br>- **클수록** 빠르게 수렴하지만 **불안정하거나 과적합** 가능|
| `iterations`    | 200, 500, 800   | 학습에 사용할 트리 수 (전체 부스팅 단계 수)             | - 너무 **작으면** 언더피팅<br>- 너무 **크면** 과적합 가능|

📊 모델 성능 비교 결과 (정렬 기준: F1-score 기준 내림차순)

| 순위 | 모델       | 스케일러     | F1-score   | Accuracy | ROC AUC    | 하이퍼파라미터                                                   |
| -- | -------- | -------- | ---------- | -------- | ---------- | --------------------------------------------------------- |
| 1  | CatBoost | quantile | **0.8414** | 0.8285   | **0.9066** | `depth=6`, `iterations=800`, `learning_rate=0.1`          |
| 2  | XGBoost  | standard | 0.8406     | 0.8280   | 0.9051     | `learning_rate=0.01`, `max_depth=10`, `n_estimators=1200` |
| 3  | XGBoost  | minmax   | 0.8406     | 0.8280   | 0.9051     | `learning_rate=0.01`, `max_depth=10`, `n_estimators=1200` |
| 4  | XGBoost  | maxabs   | 0.8406     | 0.8280   | 0.9051     | `learning_rate=0.01`, `max_depth=10`, `n_estimators=1200` |
| 5  | XGBoost  | robust   | 0.8406     | 0.8280   | 0.9051     | `learning_rate=0.01`, `max_depth=10`, `n_estimators=1200` |
| 6  | XGBoost  | none     | 0.8406     | 0.8280   | 0.9051     | `learning_rate=0.01`, `max_depth=10`, `n_estimators=1200` |
| 7  | XGBoost  | quantile | 0.8401     | 0.8272   | 0.9055     | `learning_rate=0.03`, `max_depth=8`, `n_estimators=800`   |
| 8  | CatBoost | standard | 0.8397     | 0.8267   | 0.9063     | `depth=8`, `iterations=500`, `learning_rate=0.1`          |
| 9  | CatBoost | minmax   | 0.8397     | 0.8267   | 0.9063     | `depth=8`, `iterations=500`, `learning_rate=0.1`          |
| 10 | CatBoost | maxabs   | 0.8397     | 0.8267   | 0.9063     | `depth=8`, `iterations=500`, `learning_rate=0.1`          |
| 11 | CatBoost | none     | 0.8397     | 0.8267   | 0.9063     | `depth=8`, `iterations=500`, `learning_rate=0.1`          |
| 12 | CatBoost | robust   | 0.8397     | 0.8267   | 0.9063     | `depth=8`, `iterations=500`, `learning_rate=0.1`          |


## Deep Learning Model  

이 모델은 3개의 은닉층을 갖는 다층 퍼셉트론(MLP) 구조로, 
각 층 사이에 ReLU 활성화를 적용하여 비선형성을 확보했습니다. 이진 분류 문제이므로 **손실 함수는 BCEWithLogitsLoss**를 사용하여 출력 로짓(logit) 값을 직접 처리하도록 했습니다.
**최적화 알고리즘으로는 Adam 옵티마이저**를 사용하여 학습 안정성과 수렴 속도를 향상시켰으며, 출력값을 확률로 변환하기 위해 F.sigmoid 함수를 적용하여 0~1 사이로 정규화된 예측 결과를 얻었습니다.

✅ 최종 평가 지표 (Validation Best Accuracy 기준)

| Metric    | Score      |
| --------- | ---------- |
| F1 Score  | **0.7327** |
| Precision | **0.7386** |
| Recall    | **0.7308** | 
| Accuracy  | **73.95%** |


## 📊데이터 수치 시각화

<br>

![고객이탈비율](images/graph1.png)


<br>

![고객이탈확귤분포](images/graph2.png)

<br>

![이탈확률Top10고객](images/graph3.png)

<br>

![특성별_평균값_비교](images/graph4.png)

https://youtu.be/rfy6Yky6yMs

# 결론

✅ 모델 성능 비교 순위표

| 순위 | 모델        | 스케일러     | F1-score   | Accuracy | ROC AUC | 비고                |
| -- | --------- | -------- | ---------- | -------- | ------- | ----------------- |
| 1  | CatBoost  | Quantile | **0.8414** | 0.8285   | 0.9066  | 최고 성능, 최적 파라미터 적용 |
| 2  | XGBoost   | Standard | 0.8406     | 0.8280   | 0.9051  | 거의 동급 성능          |
| 3  | CatBoost  | Standard | 0.8397     | 0.8267   | 0.9063  | 스케일러에 덜 민감        |
| 4  | MLP (딥러닝) | None     | 0.7327     | 0.7395   | —       | 기본 MLP 구조, 성능 열세  |

✅ 결론: 최종 선택 모델 : CatBoost + Quantile Scaler

📌 왜 CatBoost + Quantile Scaler가 최적의 선택인가?<br>

최고의 성능 (F1 = 0.8414, Accuracy = 82.85%, ROC AUC = 0.9066)
→ 클래스 불균형 문제에 강하고, 예측의 일관성과 안정성이 가장 뛰어남.

최적의 하이퍼파라미터 탐색 결과 적용
→ depth=6, iterations=800, learning_rate=0.1로 세밀하게 조정된 모델.

스케일러 의존도 낮음 & 처리 용이성
→ CatBoost는 범주형 처리에 강하고, Quantile 스케일러와의 조합에서도 효과적임.

⚠️ 딥러닝 모델은 왜 성능이 낮았나?

| 항목       | 설명                                     |
| -------- | -------------------------------------- |
| 데이터양 부족  | 딥러닝은 많은 데이터에서 강력한 성능을 보이지만, 현재는 다소 제한적 |
| 구조 단순    | 은닉층 3개 + 기본 MLP 구조는 복잡한 패턴 학습에 한계      |
| 정규화 미흡   | 정규화 또는 스케일링이 적용되지 않아 학습 안정성 저하 가능성     |
| 오버피팅 가능성 | epoch 수 100으로 고정되어 오버피팅 우려도 존재         |


# 이탈률 감소 대책

### 1. TV & 영화 번들 서비스 유도
분석 근거: 인터넷을 포함한 TV와 Movie를 둘 다 구독할 때 이탈률 감소에 유의미한 변화가 나타남.

대책:

- TV/영화 패키지 결합 시 할인 제공

- 신규 고객 대상 TV & 영화 무료 체험 제공

- 구독 장기 유지 시, 사용자가 주로 시청한 분야의 시사회 티켓 제공

### 2. 계약 유도 전략
분석 근거: remaining_contract가 길게 남아 있을 때, 이탈률이 낮음

대책:
- 계약 만료 고객에게 리턴 쿠폰 제공 (1달 무료 이용권)

- 계약 기간 만료 직전 자동 리마인드(문자, 메일링) 및 특별 혜택(계약 연장 시, 사용자 특화 오프라인 아이템 제공 - 영화 & 뮤지컬 티켓, 방청권등)

### 3. 다운로드 용량별 맞춤 대응 전략
분석 근거 : download data 가 적정량 있을 경우 이탈률이 낮지만 download over limit이 많을 경우 이탈률이 높은 경향을 보임
대책 :
- 데이터 사용량에 따른 고객 세분화 대응

| 고객 유형     | 특징                                                | 이탈률 | 행동 패턴              |
| --------- | ------------------------------------------------- | --- | ------------------ |
| 일반 사용자    | 보통 수준의 다운로드 사용                                    | 낮음  | 꾸준히 사용, 이탈 가능성 낮음  |
| ** 헤비유저** | 평균보다 훨씬 높은 `download_avg` & `download_over_limit` | 높음  | 초과요금에 민감, 불만족 시 이탈 |

- 헤비사용자를 위한 무제한 요금제 제안

- 개인 & 사업자 요금제 분리 및 혜택 제공

- 사용량 초과 전 알림 & 제어 시스템 강화

  - 일정 한계치 접근 시 사전 메시지 자동 발송
      한도 초과 시 자동으로 데이터 속도 제한 or 추가 사용 동의 팝업

### 4. 1인가구 특별 혜택 전략
분석 근거 : ID는 계약 건의 총 인원 수를 포함하는 것이 아닌, 계약 1건당 카운트가 올라간다.
- 가구 내의 인원 수 보다 계약한 가구의 총 수가 중요하다.

대책 : 나날히 늘어가는 1인 가구를 위한 특별 혜택을 제공한다.
- 사회 초년생을 위한 무제한 데이터 업로드 서비스 제공.
- 인터넷 계약 시, 초기 비용 환급 서비스

# 한 줄 회고

| 이름     | 회고 내용 |
|----------|-----------|
| 김재아   | 첫 팀플에 이어 두번째도 팀원들을 정말 잘 만나서 호다닥 해치웠습니다..! 머신러닝도 열심히 해보고 딥러닝 부분도 팀원분들의 도움이 컸습니다! |
| 박빛나   | 수업을 하면서 어려웠던 머신러닝부분을 지원하면서 많이 배울 수 있는 시간이었다. 특히 딥러닝 부분은 팀원들의 도움이 매우 컸다! 이탈하지말아조... 영원히... 우린... 한팀.... |
| 이승철   | 팀프로젝트하면서 얘들은 왜 여기서 배우고 앉아있나 하는 생각. 선출도 아니고 현역같은 실력과 재주 그리고 근사한 인성까지 가지고 있으면서. |
| 하종수   | 분업화해서 진행했더니 일의 진행이 너무 잘되었다...! 이거시 팀플..? 팀원들이 너무 좋아요! |
| 한성규   | 절 이끌어주시고, 제가 볼 수 없는 시각에서 다양한 의견을 제시해주신 팀원분들에게 무한한 감사를... 연휴 때 쉬고, 할 때 하는 아름다운 팀이었고, 프로젝트 기간동안 행복했습니다~🥰 |








