# 1. 데이터 수집 및 로드

```python
df = pd.read_csv('internet_service_churn.csv')

```
# 2. 데이터 구조 및 변수 이해

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 72274 entries, 0 to 72273
    Data columns (total 11 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   id                           72274 non-null  int64  
     1   is_tv_subscriber             72274 non-null  int64  
     2   is_movie_package_subscriber  72274 non-null  int64  
     3   subscription_age             72274 non-null  float64
     4   bill_avg                     72274 non-null  int64  
     5   reamining_contract           50702 non-null  float64
     6   service_failure_count        72274 non-null  int64  
     7   download_avg                 71893 non-null  float64
     8   upload_avg                   71893 non-null  float64
     9   download_over_limit          72274 non-null  int64  
     10  churn                        72274 non-null  int64  
    dtypes: float64(4), int64(7)
    memory usage: 6.1 MB
    None
    (72274, 11)


- 행/열 수 : 72274행 11열
- float 4개, int 7개
- 고유 식별값1개, 이진형 3개, 수치 범주형 2개, 연속형 5개
- 타겟 : churn




 churn 0 / 1 양 비교

` 0 (32224)  1 (40050)`


| No. | 컬럼명                        | 설명 |
|-----|------------------------------|------|
| 0   | id                           | 고유 구독자 ID |
| 1   | is_tv_subscriber             | TV 구독 여부 |
| 2   | is_movie_package_subscriber | 영화 패키지 구독 여부 |
| 3   | subscription_age            | 구독 기간 |
| 4   | bill_avg                    | 지난 3개월 평균 청구 금액 |
| 5   | reamining_contract          | 계약의 남은 연수 (null일 경우 계약 없음, 계약 중 취소 시 위약금 발생) |
| 6   | service_failure_count       | 지난 3개월간 서비스 장애로 인한 콜센터 고객 통화 횟수 |
| 7   | download_avg                | 지난 3개월 인터넷 사용량 (GB) |
| 8   | upload_avg                  | 지난 3개월 평균 업로드 (GB)<br>(최근 3개월 다운로드 및 업로드 평균 사용량은 3GB 제한 내에 있음) |
| 9   | download_over_limit         | 지난 9개월 동안 다운로드 제한 초과 횟수<br>(초과 시 추가 요금 발생) |
| 10  | churn                       | 서비스 취소 여부 |

# 3.데이터 요약 및 통계


                     id  is_tv_subscriber  is_movie_package_subscriber  \
    count  7.227400e+04      72274.000000                 72274.000000   
    mean   8.463182e+05          0.815259                     0.334629   
    std    4.891022e+05          0.388090                     0.471864   
    min    1.500000e+01          0.000000                     0.000000   
    25%    4.222165e+05          1.000000                     0.000000   
    50%    8.477840e+05          1.000000                     0.000000   
    75%    1.269562e+06          1.000000                     1.000000   
    max    1.689744e+06          1.000000                     1.000000   
    
           subscription_age      bill_avg  reamining_contract  \
    count      72274.000000  72274.000000        50702.000000   
    mean           2.450051     18.942483            0.716039   
    std            2.034990     13.215386            0.697102   
    min           -0.020000      0.000000            0.000000   
    25%            0.930000     13.000000            0.000000   
    50%            1.980000     19.000000            0.570000   
    75%            3.300000     22.000000            1.310000   
    max           12.800000    406.000000            2.920000   
    
           service_failure_count  download_avg    upload_avg  download_over_limit  \
    count           72274.000000  71893.000000  71893.000000         72274.000000   
    mean                0.274234     43.689911      4.192076             0.207613   
    std                 0.816621     63.405963      9.818896             0.997123   
    min                 0.000000      0.000000      0.000000             0.000000   
    25%                 0.000000      6.700000      0.500000             0.000000   
    50%                 0.000000     27.800000      2.100000             0.000000   
    75%                 0.000000     60.500000      4.800000             0.000000   
    max                19.000000   4415.200000    453.300000             7.000000   
    
                  churn  
    count  72274.000000  
    mean       0.554141  
    std        0.497064  
    min        0.000000  
    25%        0.000000  
    50%        1.000000  
    75%        1.000000  
    max        1.000000  


구독 기간 최소값이 음수

# 4.결측치 및 이상치 탐색 , 8. 데이터 전처리



 - 고유 식별값인 id 컬럼 삭제

 - reamining contract 특성 remaining contract의 오타로 판단.

 - .rename 사용해서 컬럼명 변경

변경된 컬럼명 확인


    Index(['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
           'bill_avg', 'remaining_contract', 'service_failure_count',
           'download_avg', 'upload_avg', 'download_over_limit', 'churn'],
          dtype='object')


### 이상치 확인 & 제거


describe에서 구독기간인 'subscription_age' 항목에서 최소값이 음수인 것을 확인.


 subscription_age 컬럼의 최소값 확인


`-0.02`



![Subscription Age](/images/subscription_age_outlier.png)




` 이상치 제거 DataFrame 갱신`

` 이상치 제거 후 행 수 반환 -> 72273`

 `subscription_age 컬럼의 최소값 확인 -> 0.0`

    72273
    0.0


### 결측치 처리


    is_tv_subscriber                   0
    is_movie_package_subscriber        0
    subscription_age                   0
    bill_avg                           0
    remaining_contract             21572
    service_failure_count              0
    download_avg                     381
    upload_avg                       381
    download_over_limit                0
    churn                              0
    dtype: int64



 - remaining contract 결측치 21572

 - download_avg, upload_avg 결측치 381

remaining_contract = null -> 잔여 계약 기간 x

현재 값의 기준은 year 단위,

1년을 365로 나눌 경우 약 0.002739 == 약 0.00274 --> 소수점 2자리까지 표기가 불가능한 day 단위의 남은 계약의 경우 0으로 표기되었을 가능성

1일을 0.00274로 추측할 경우 0.01은 대략 3.6일 즉 4일 정도로 예측 가능

그러므로 현재 측정값 약 4일 이상의 계약이 남았을 경우만 0.01 이상의 값으로 표기하고 그 이하는 0으로 표기해놨다고 추측,

0값으로 표기된 관측값은 3일 미만으로 계약이 남은 경우라고 판별하고 null인 값들을 진짜 계약이 만료된 경우로 판별하는 것이 조금 더 합리적인 판단,

하지만 더 분석해본 결과 0으로 관측되는 값들의 churn이 거의 모두 1 (이탈)로 분류 되는 것으로 보아 그냥 계약 만료로 봐도 무방할 것으로 판단


    이탈 x 잔여계약 null:  1853
    이탈 x 잔여계약 0 :  72
    이탈 o 잔여계약 0 :  16291


#### 이탈 x . 잔여 계약 기간  null -> 0으로 처리



`-  dl avg ul avg도  72,273개 중 381개 0.53%로 모델 학습에 영향 거의 없을것으로 판단`
`-  remaining_contract, dl_avg, ul_avg null 값 0으로 채우기`
`- .fillna(0) 사용하여 결측값 채우기.`


    is_tv_subscriber               0
    is_movie_package_subscriber    0
    subscription_age               0
    bill_avg                       0
    remaining_contract             0
    service_failure_count          0
    download_avg                   0
    upload_avg                     0
    download_over_limit            0
    churn                          0
    dtype: int64



# 5. 변수 분포 시각화 6. 변수 간 관계 시각화

### 이진형 변수 vs Churn

![subscriber_countplot](/images/subscriber_countplot.png)
    


### 수치 범주형 변수 vs Churn

#### download_over_limit
![download_over_limit](/images/download_over_limit_countplot.png)
    


### 연속형 변수 vs Churn

#### subscribtion age viloin plot

![subscribtion age viloin plot](/images/subscribtion_age_viloinplot.png)

구독 기간 2년까지에서 이탈 ↑
이후 기간에서 ↓

#### download & upload avg.

    
![download & upload avg](/images/download_upload_avg.png)
    


이탈고객 대부분이 사용량이 거의 없음.

#### remaining contract viloin plot


![remaining contract viloin plot](/images/remaining_contract_viloinplot.png)

이탈고객 대부분 잔여 계약 X

### 컬럼을 묶어 시각화

#### download_avg, upload_avg 묶어서 시각화

데이터가 x축인 download_avg 1000, y축인 upload_avg 200 안에 많이 모여있어서 잘 나타나지 않는 것 같음

    
![download_avg_upload_avg_combine](/images/download_avg_upload_avg_combine.png)
    


upload와 download 모두 없거나 적은 회원의 이탈이 몰려있음.

0을 벗어난 구간의 회원 이탈률이 낮은 경향

# 07.상관관계 및 교차분석

![heatmap1st](/images/heatmap1st.png)
    


#### 상관관계가 있는 주요 특성

- target인 churn과 양의 상관관계 :  download_over_limit(0.16)

- target인 churn과 음의 상관관계 : remaining_contract(-0.63), is_tv_subscriber(-0.33), is_movie_package_subscriber(-0.31), download_avg(-0.30), upload_avg(-0.16) subscription_age(-0.12)

# 9. Feature Engineering

- 현재 시각화 분석시 TV 구독, 영화 구독에 대해 각각 이탈 or 유지만 분석함,
- 하지만, TV 구독과 영화 구독을 둘 다 하지 않는 경우, 둘 다 하는 경우는 알 수 없기에,
- 
0. : 구독 없음
1. : TV 구독
2. : 영화 구독
3. : 둘 다 구독


과 같은 형태로 새로 특성을 만들어 상관성을 찾고 시각화하여 확인하는게 좋지 않을까 라는 의문에서 새로운 컬럼 'subscription_status' 생성

- 시각화로 이탈률 확인



    
![subscribe_combine_rate.](/images/subscribe_combine_rate.png)
    


두 항목 모두 구독한 구독자의 이탈률이 현저히 낮은 것을 확인.

None에서의 이탈률도 높았지만, 영화만 구독한 구독자의 이탈률이 1로 나타난 것이 이상하여 항목별 이탈자 수로 확인.

-  시각화로 이탈자 수 확인


    
![subscribe_combine_count](/images/subscribe_combine_count.png)
    


이탈자 수 시각화 했을 하나도 구독을 안한 집단과 둘 다 구독한 집단의 이탈자 수 차이가 확연하게 보임.

영화만 구독하는 사람들은 그래프에서 보이지 않음.

정확한 이탈자 수 확인을 위해 텍스트로 확인


` 이탈 여부와 함께 그룹별 고객 수 확인`


    churn                          0      1
    subscription_status_label              
    Both                       15990   8193
    Movie only                     0      2
    None                        1386  11963
    TV only                    14848  19891


영화만 단일로 구독하는 사람은 2명

대부분의 고객은 TV만 구독하거나, TV + 영화 패키지를 함께 사용하고 있음

movie only를 선택한 고객은 72,000명 중 2명 → 0.003% 수준


```python
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap with subscription status")
plt.show()
```


    
![heatmap2](/images/heatmap2nd.png)
    


TV와 영화 구독자들을 묶은 새로운 컬럼인 subscription에서

기존 TV 구독과 영화구독 단일 항목에서의 상관관계보다 높은 0.37의 음의 상관관계가 나타남을 확인.

# 10. 최종 요약 및 인사이트 도출

## 데이터 요약 및 인사이트

- TV와 영화를 구독중인 고객의 이탈률이 상대적으로 낮음 → 부가서비스 제공이 이탈 방지에 효과적임을 알 수 있음.

- remaining_contract가 있을 때 이탈률이 낮음 -> 계약기간 도중 해약 시 패널티가 있는 현 제도가 계약 유지에 효과적으로 작용하고 있음

- download_avg 와 이탈률의 음의 상관관계 -> 데이터 사용량이 이탈 방지에 작용

### 향후 분석/모델링 방향

- 군집 분석: 고객 데이터 사용량 세분화를 통한 타겟

- 서비스 품질 지표 강화 & 수정: remaining_contract, is_tv_subscriber & is_movie_package_subscriber를 묶은 subscription status 등과 이탈의 연관성 모델링 강화, 기존 컬럼 삭제 고려

- remaining_contract 의 상관관계가 다른 항목들보다 확연히 큼. 이후 학습과정에서 remaining contract만으로도 학습이 되는 상황이 초래될 수 있어서 주의해야함.

## 이탈률 감소 대책

### 1. TV & 영화 번들 서비스 유도
분석 근거: 인터넷을 포함한 TV와 Movie를 둘 다 구독할 때 이탈률 감소에 유의미한 변화가 나타남.

대책:

- TV/영화 패키지 결합 시 할인 제공

- 신규 고객 대상 TV & 영화 무료 체험 제공

- 구독 장기 유지 시, 사용자가 주로 시청한 분야의 시사회 티켓 제공

### 2. 계약 유도 전략
분석 근거: 잔여 계약 기간이 있을 때 이탈률이 낮음

대책:

- 무계약 고객 대상으로 장기 계약 유도 + 사은품/할인 제공

- 계약 기간 만료 직전 자동 리마인드(문자, 메일링) 및 특별 혜택(계약 연장 시, 사용자 특화 오프라인 아이템 제공 - 영화 & 뮤지컬 티켓, 방청권등)

### 3. 다운로드 용량별 맞춤 대응 전략
분석 근거 : download data 가 적정량 있을 경우 이탈률이 낮지만 download over limit이 많을 경우 이탈률이 높은 경향을 보임

대책 :
- 데이터 사용량에 따른 고객 세분화 대응

| 고객 유형     | 특징                                                | 이탈률 | 행동 패턴              |
| --------- | ------------------------------------------------- | --- | ------------------ |
| 일반 사용자    | 보통 수준의 다운로드 사용                                    | 낮음  | 꾸준히 사용, 이탈 가능성 낮음  |
| ** 헤비유저** | 평균보다 훨씬 높은 `download_avg` & `download_over_limit` | 높음  | 초과요금에 민감, 불만족 시 이탈 |

-  초과요금 발생 고객 분석을 기반으로 무제한 요금제 / 패키지 리디자인
- 사용량 초과 전 알림 & 제어 시스템 강화
    : 일정 한계치 접근 시 사전 경고 메시지 자동 발송
    한도 초과 시 자동으로 데이터 속도 제한 or 추가 사용 동의 팝업 → 불만 예방


### 4. 1인가구 특별 혜택 전략
분석 근거 : ID는 계약 건의 총 인원 수를 포함하는 것이 아닌, 계약 1건당 카운트가 올라간다.

가구 내의 인원 수 보다 계약한 가구의 총 수가 중요하다.
대책 : 나날히 늘어가는 1인 가구를 위한 특별 혜택을 제공한다.

사회 초년생을 위한 무제한 데이터 업로드 서비스 제공.

