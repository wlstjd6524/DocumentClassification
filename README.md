[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hoEJPLZy)

## 👨‍🏫 프로젝트 소개
## Document Type Classification Competition | 문서 타입 분류 대회 <br> <br>
## Team
<table>
  <tr>
    <td> <div align=center> 👑 </div> </td>
    <td> <div align=center> 🙍 </div> </td>
    <td> <div align=center> 🙍 </div> </td>
    <td> <div align=center> 🙍‍♂ </div> </td>
    <td> <div align=center> 🙍‍♂ </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <b>이진성</b> </div> </td>
    <td> <div align=center> <b>박세희</b> </div> </td>
    <td> <div align=center> <b>서효림</b> </div> </td>
    <td> <div align=center> <b>유창준</b> </div> </td>
    <td> <div align=center> <b>이건우</b> </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/acd706d5-ec12-46a5-9c00-5c4db3e9487e" /> </td> <!--이진성-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/4c235a37-381f-429b-b71e-294a687bca87" /> </td> <!--박세희-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/93acc3ec-1eb9-457a-a731-db36b1f5c6d0" /> </td> <!--서효림-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/6bc69b86-aef3-4b23-a870-709324ca26b7" /> </td> <!--유창준-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/00940462-ec40-4b50-9942-395ec124d2e1" /> </td> <!--이건우-->
  </tr>
  <tr>
    <td> <div align=center> <a href="https://github.com/wlstjd6524"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    </tr>
</table>

## 💻 개발환경 및 도구
- Python 3.10.0
- Linux OS by Upstage GPU Server
- PyTorch
- timm
- Albumentations
- OpenCV (cv2)
- Numpy
- Pandas
- Scikit-learn
- tqdm


## 📏 프로젝트 목적
실제 산업 현장에서 발생하는 문서 데이터는 금융, 의료, 보험, 물류 등 도메인을 가리지 않고 존재하며 많은 회사에서 아날로그 데이터의 디지털화를 희망하고 있습니다.
이를 위해 아날로그 문서 데이터의 종류를 식별하고자 합니다.

본 프로젝트는 다양한 유형의 문서 이미지를 17개 클래스로 분류하는 이미지 분류 문제를 해결하는 것을 목표로 합니다.
또한 분류 문제와 딥러닝의 전체적인 파이프라인 경험, 그리고 PyTorch 의 프레임워크의 입문을 목표로 딥러닝의 전체적인 Flow 이해를 돕기 위해 진행되는 프로젝트 입니다.

문서 특성상 발생하는 '회전(skew), 종횡비(Aspect Ratio)의 차이, 해상도 불균형' 등의 문제에 강건한 모델을 구축하고자 하였습니다.
또한 단일 모델 성능의 한계를 극복하기 위해 다른 특성을 가진 두 개의 강력한 백본 모델 ConvNext-Base 와 Swin-Base 를 각각 학습한 뒤, 테스트 데이터에 대해 logits 단위 앙상블(Logit Ensemble) 을 수행하여 최종 예측 성능을 극대화 하는 작업을 진행하였습니다.

실험 결과를 기반으로 재현 가능하고 확장 가능한 추론 파이프라인을 구축하고자 하였습니다.


## 📁 프로젝트 구조
```text
Project/
├─ data/
│     ├─ train/                 # 학습용 이미지(JPG)
│     ├─ test/                  # 테스트 이미지(JPG)
│     ├─ meta.csv
│     ├─ sample_submission.csv
│     └─ train.csv
│
├─ eda/
│     ├─ CJ_up_cv_contest_EDA.ipynb
│     ├─ geonwoo_baseline_code_for_linux/
│     ├─ jinsung_baseline_EDA/
│     └─ SH_EDA Result.pdf
│
├─ convnext_base1/
│     └─ predict_logits.pt      # ConvNeXt 테스트 logits
│
├─ swin_base_384_v1_infer_add_tta/
│     └─ predict_logits.pt      # Swin 테스트 logits
│
├─ Submission/
│     └─ submission_conv0.70_swin0.30.csv (예시)
│
├─ ensemble_convheavy.py     # 가중 앙상블 스크립트
│
├─ convnext_base_single.py   # ConvNeXt 학습 + 추론
│
└─ swin_base_384_v1_infer_add_tta.py  # Swin 학습 + 추론
```


## 🔨 프로젝트 시스템 아키텍처
<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/dfde8ab9-951d-4713-bee4-d92323ee4ea7" />


# 📷 Data 정보
- 학습 데이터  : 총 1570장의 문서 이미지가 존재하며 17개 클래스 안에 각 클래스 별로 46 ~ 100 장 이미지가 랜덤으로 분포되어 있습니다.
  - train.csv : 학습 이미지의 이름과 클래스 사의 mapping 정보 | ID, Target
  - meta.csv  : 클래스 이름과 인덱스의 mapping 정보 | target, class_name
  - train/    : 학습 이미지가 분포되어 있음

- 테스트 데이터 : 총 3140장의 문서 이미지가 존재하며 여러 Augmentations 이 적용돼있음, 어떤 Augmentations 는 Secret
  - sample_submission.csv : 예측값을 채워넣을 더미 파일
  - test/                 : 테스트 이미지가 분포되어 있음
 

## 🧾 Class 정보
- 총 17개 클래스가 존재합니다.
```
| target | class_name                                            | Mapping Korea 
| -----: | ----------------------------------------------------- | ------------- 
|      0 | account_number                                        | 계좌 번호      
|      1 | application_for_payment_of_pregnancy_medical_expenses | 임신 의료비 지불 신청서
|      2 | car_dashboard                                         | 자동차 대시보드
|      3 | confirmation_of_admission_and_discharge               | 입원 및 퇴원 확인서
|      4 | diagnosis                                             | 진단서
|      5 | driver_lisence                                        | 운전면허증
|      6 | medical_bill_receipts                                 | 의료비 영수증
|      7 | medical_outpatient_certificate                        | 외래진료 증명서
|      8 | national_id_card                                      | 주민등록증
|      9 | passport                                              | 여권
|     10 | payment_confirmation                                  | 결제 확인서
|     11 | pharmaceutical_receipt                                | 약품 영수증
|     12 | prescription                                          | 처방전
|     13 | resume                                                | 이력서
|     14 | statement_of_opinion                                  | 소견서
|     15 | vehicle_registration_certificate                      | 차량 등록증
|     16 | vehicle_registration_plate                            | 차량 등록 번호판
```


## 📒 EDA
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/9855738d-99bb-4d50-8e60-c15bdf38e54c" /> <br> <br>
각 클래스 별로 분포도가 어느정도인지, 해상도는 보통 어느 곳을 주로 바라보는지, Train 과 Test 데이터의 품질 차이 등등
여러가지 EDA 를 진행하였습니다.
(파악해본 그래프 형식 이미지가 너무 많아 발표자료에서 대체하겠습니다.)


## ✍ 구현 기능
### 0. 데이터 전처리 및 라벨 정합성 보정
- train.csv 내 일부 샘플의 잘못된 라벨을 사전 분석 후 수동으로 교정하게 되었습니다.
```
# 오라벨 List {
        "45f0d2dfc7e47c03.jpg": 7, "aec62dced7af97cd.jpg": 14, "0583254a73b48ece.jpg": 10,
        "1ec14a14bbe633db.jpg": 7, "c5182ab809478f12.jpg": 14, "8646f2c3280a4f49.jpg": 3,
        "38d1796b6ad99ddd.jpg": 10,
}
```
- 학습 시잔 전 Label Correction 로직을 통해 자동으로 수정된 CSV 를 생성하였습니다.
- 또한 모든 모델에서 동일한 Corrected Label 을 사용하여 학습 일관성을 확보하였습니다.


### 1. 모델 학습 (5-Fold Stratifed Cross Validation)
- 학습전략으로 StratifiedKFold(n_splits = 5) 를 적용하였습니다.
- 클래스 불균형 문제 해결을 위하여 class weight 기반 Validation Loss 를 계산하여 진행하였습니다.
- 각 학습마다 Early Stopping 을 적용하였습니다.
  - Swin : Patience 7
  - ConvNext : Patience 5
- 그 후 각 fold 별 Best ChekcPoint 를 저장하는 구조로 진행하였습니다.


### 2. 모델 선정
- 선정모델 : ConvNext 
- 모델 선정 이유 : CNN 계열의 강력한 지역 패턴 학습 능력 과 문서 이미지의 텍스트/레이아웃 특징에 강함
- 선정모델 : Swin Transformer
- 모델 선정 이유 : 윈도우 기반 self-attention 으로 전역 문맥 파악에 유리하며 고정 해상도(384) 에서 안정적인 학습 특성을 가짐
- 서로 다른 구조 Comment : 따라서 서로 강점을 가지면서 구조가 다른 (CNN + Transformer) 의 조합으로 앙상블 효과 극대화
- 모델 학습 구조 : 모델 학습 구조는 다음을 따름
  - 5-Fold Stratified Cross Validation
  - AMP 기반 Mixed Precision Training
  - CosineAnnealingWarmRestarts Scheduler
  - Early Stopping
  - rot90 TTA + fold logits 평균 
- 평가 지표 : Macro F1-Score


### 3. 데이터 증강(Augmentation) 전략
- Swin-Base 384
  - 입력 크기 : 384 * 384 고정
  - Albumentations 기반 증강을 진행하였습니다.
    - ShiftScaleRotate (소규모 변형)
    - RandomBrightnessContrast
    - CoarseDropout
    - Normalize + ToTensroV2

- ConvNeXt-Base
  - 문서 이미지 특성을 고려한 고급 전처리를 진행하였습니다.
    - Deskew : 문서 기울기 자동 및 섬세한 보정
    - Letterbox Reszie : 종횡비 유지 + 패딩
  - Aspect Ratio Bucketing
    - 이미지 비율에 따라 해상도 버킷을 생성하여 왜곡을 최소화 하였습니다.
  - Albumentations 증강
    - ShiftScaleRotate
    - RandomBrightnessContrast
    - Blur 계열 증강
    - CoarseDropout


### 4. Mixup 기반 학습 안정화
- 두 모델 모두 timm Mixup 을 적용하였습니다.
- SoftTargetCrossEntropy 를 사용하였습니다.
- Label smoothing 을 적용하였고 과적합 방지 및 일반화 성능 향상을 기대할 수 있었습니다.


### 5. 추론(TTA) 및 Fold Ensemble
- rot90 기반 Test Time Augmentation 을 부여하였습니다.
  - 0, 90, 180, 270 회전 을 부여
- 각 fold 의 Logits 평균을 도출하였습니다. -> Fold Ensemble
- 모데별 최종 테스트 Logits 를 생성하였습니다.


### 6. Logit Ensemble 기반 최종 제출 생성
- ConvNext / Swin 각각의 Predict_logits.pt 를 로드하여 logits 단위 가중합을 수행하였습니다.
```
final_logits = w_conv * conv_logits + (1 - w_conv) * swin_logits
```
- softmax 없이 argmax(final_logits) 로 최종 예측을 진행하였고 가중치 실험 결과 Conv:Swin 비율을 0.70:0.30 로 섞어서 추론한 결과값이 안정적인 최고 성능을 내는 걸 확인 할 수 있었습니다.


## 🚨 문제 및 인사이트 도출
### 1. 로컬 성능 대비 제출 성능 하락 문제
#### 문제
로컬 환경에서 추론 결과를 확인했을 때는 분류가 비교적 잘 수행되는 것으로 보였으나, 실제 리더보드에 제출했을 때 점수가 기대보다 낮게 측정되는 문제가 발생하였다.
즉, 로컬 성능과 제출 성능 간의 괴리가 지속적으로 관측되었다.

#### 해결
원인 분석 결과, 입력 이미지의 해상도와 비율이 제각각이라는 점이 주요 문제로 판단되었습니다.
이에 따라 모델이 이미지 크기나 레이아웃 변화에 덜 민감하도록 하기 위해 다음과 같은 시도를 진행하였습니다.

- 패치 단위 이동(shift window) 기반의 특징 추출 방식 고려
- 이미지 크기 변화에 강건한 학습 구조 설계
- Triplet Loss를 통한 클래스 간 거리 학습을 시도하였으나, 시간 제약으로 인해 최종 적용에는 실패

#### 인사이트
문서 이미지 분류 문제에서는 단순한 분류 정확도보다, 입력 이미지 스케일과 레이아웃 변화에 대한 모델의 강건성이 실제 성능에 큰 영향을 미친다는 점을 확인하였습니다.

---

### 2. 문서 클래스별 식별력 향상에 대한 고민
#### 문제
문서 클래스 간 유사성이 높아, 특히 세부 문서 유형을 구분하는 데 있어 모델의 식별력이 충분하지 않다고 판단되었습니다.
또한 어떤 방식으로 데이터를 가공해야 모델이 문서를 더 잘 이해할 수 있을지에 대한 고민이 지속되었습니다.

#### 해결
문제 해결을 위해 다음과 같은 접근을 시도하였습니다.
- 입력 해상도를 높이는 전략 : 해상도를 높이면 글자 정보가 더 선명하게 반영되어 문서 간 차이를 더 잘 학습할 수 있을 것으로 기대
- Canny Edge Detection 이미지 활용 : 원본 이미지와 함께 edge 정보를 학습 및 TTA에 활용하는 방식을 실험하였습니다.
  - 일부 성능 개선 가능성은 보였지만, 연산 비용과 추론 시간이 과도하게 증가하여 최종 적용은 포기하였습니다.

### 인사이트
문서 분류 문제에서는 텍스트의 형태와 배치가 핵심 정보이므로, 해상도는 성능에 직접적인 영향을 미치는 중요한 요소임을 확인하였습니다.
다만, 성능 향상과 실용성(속도, 자원) 사이의 균형이 반드시 고려되어야 함을 체감하였습니다.

---

### 3. 모델 선택의 어려움
#### 문제
이미지 분류에 사용할 수 있는 모델의 종류가 매우 다양하고,
각 모델마다 크기, 파라미터 수, 학습 특성이 달라 본 대회에 가장 적합한 모델을 선택하는 과정 자체가 어려움으로 작용하였습니다.

#### 해결
단일 모델에 의존하기보다는 다양한 모델을 직접 실험해보는 전략을 선택하였습니다.
- 여러 모델을 실제로 학습·추론해보며 장단점을 체감
- 초기에는 hierarchy 구조를 가진 모델을 실험했으나 성능 향상이 제한적
- 이후 ConvNeXt 계열과 Swin Transformer 계열을 앙상블하는 방향으로 전환

#### 인사이트
각 모델은 서로 다른 강점을 가지고 있으며,
단일 모델 최적화보다 서로 다른 구조(CNN + Transformer)를 결합하는 앙상블 전략이 더 효과적일 수 있다는 점을 경험적으로 확인하였습니다.
또한, 2개 이상의 모델을 실제 프로젝트에서 병합해본 첫 경험으로서 큰 학습 효과를 얻을 수 있었습니다.

---

### 4. 문서 분류 기준 설정의 어려움 (레이아웃 vs 제목)
#### 문제
문서를 분류할 때 전체 레이아웃을 기준으로 볼지, 문서 상단의 제목을 기준으로 볼지 명확한 기준을 잡기 어려웠습니다.
특히 다음과 같이 레이아웃과 텍스트 구조가 거의 동일한 클래스들에서 오분류가 빈번하게 발생하였습니다.

- 입퇴원확인서 / 입통원확인서 / 진료확인서
- 진단서 / 소견서

한 글자 차이(퇴/통, 진/소)가 핵심 결정 요소였으나,
CNN 기반 전역 특징 학습만으로는 이러한 미세한 차이를 충분히 반영하기 어려웠습니다.

#### 해결
문서 상단 제목 영역에 주목하도록 Title-focused augmentation을 적용하였습니다.

- 혼동이 잦은 클래스 쌍(3/7, 4/14)에 대해 집중 분석
- 혼동 행렬과 오분류 이미지를 반복적으로 확인
- 모델이 실제로 어떤 영역을 보고 판단하는지에 대한 분석 실험 진행

#### 인사이트
문서 분류 문제에서는 모델의 크기보다 “어디를 보게 할 것인가”가 훨씬 중요하다는 점을 깨달았습니다.
특히 의미적·형태적 유사성이 높은 문서일수록, 관심 영역을 명확히 유도하는 전략이 필수적임을 확인하였습니다.

---

### 5. 로컬 성능과 리더보드 성능 간 극심한 괴리
#### 문제
로컬 검증에서는 매우 높은 F1-score(Local ≈ 0.95)를 기록했으나,
리더보드 제출 시 성능이 크게 하락(Local 대비 약 0.58)하는 문제가 발생하였습니다.

#### 해결
문제의 원인을 다음과 같이 분석하였습니다.

- 과도한 규칙 기반 보정
- 지나치게 강한 Augmentation
- 데이터 분포에 맞지 않는 Bias 적용

이에 따라 복잡한 규칙을 제거하고, 단순하지만 안정적인 학습 구조로 회귀하는 방향으로 수정하였습니다.

#### 인사이트
비록 최종 성능 개선으로 이어지지는 않았지만,
과도한 개입이 오히려 일반화 성능을 해칠 수 있다는 점을 직접 실험을 통해 확인한 경험은 중요한 학습 과정이었습니다.

---

### 6. 이미지 전처리 접근 방식에 대한 어려움
#### 문제
프로젝트 초반에는 이미지 전처리에 대한 경험이 부족하여,
3차원 데이터(이미지)를 어떤 방식으로 가공해야 모델이 이해하기 쉬운 형태가 되는지에 대한 방향 설정이 어려웠습니다.

#### 해결
다음과 같은 접근을 통해 문제를 해결하고자 하였습니다.

- 각 모델의 구조와 강점을 먼저 이해하는 것부터 시작
- 이미지가 모델 내부에서 어떤 방식으로 분할되고 처리되는지에 주목
- 프로젝트 일정과 목표를 고려하여, 지나치게 세부적인 이론 학습보다는 “어떤 모델이 어떤 상황에서 강한가”를 중심으로 학습 방향 설정
- 데이터를 직접 눈으로 많이 확인하며 대회의 본질을 이해하려는 노력 병행

#### 인사이트
모델 자체보다도, 데이터를 어떻게 바라보고 이해할 것인가에 대한 관점 설정이 매우 중요하다는 점을 체감하였습니다.
특히 딥러닝에서는 데이터 이해가 곧 성능으로 직결됨을 깨닫게 되었습니다.

---

### 7. K-Fold 신뢰성 및 일반화 문제
#### 문제
로컬에서는 높은 점수가 안정적으로 측정되었으나,
리더보드 제출 시 약 0.2 수준의 성능 하락이 반복적으로 발생하였습니다.

#### 해결
다음과 같은 가설을 세우고 실험을 진행하였습니다.

- 모델이 학습 데이터를 과도하게 암기했을 가능성
- 리더보드 평가 데이터에 숨겨진 분포 차이 존재 가능성

이에 따라 학습 전략을 두 가지로 나누어 실험하였습니다.

- Case 1: 기본 Augmentation 기반의 일반적인 학습
- Case 2: Test 데이터 분포를 모사하기 위해 더 강한 Augmentation을 일부러 적용한 학습
- K-Fold 외에 Hold-out 검증도 병행하여 성능 안정성 확인

#### 인사이트
눈에 띄는 성능 향상은 없었지만,
로컬 검증 점수만으로 모델의 일반화 성능을 판단하는 것은 위험할 수 있다는 점을 깊이 이해하게 되었다.
다양한 검증 전략을 직접 시도해본 경험 자체가 프로젝트에서 중요한 자산이 되었다.

---


## 📈 결과

### Leader Board
<img width="980" height="376" alt="Image" src="https://github.com/user-attachments/assets/d0e38151-5409-4462-81a3-17962a6f2b35" />

Rank 3 🥉


## 📚 Presentation
- [발표자료]( )


## 🔎 프로젝트 한계 및 개선사항
### 1. 형태·의미적으로 유사한 문서 클래스 간 혼동 문제 (Class 3 / 7 / 14)
### 한계

프로젝트 진행 과정에서 Class 3, 7, 14 간의 오분류가 반복적으로 관측되었습니다.
특히 모델이 7번 클래스로 강하게 예측했으나 실제 정답이 3번인 경우, 혹은 그 반대의 사례가 빈번하게 발생하였습니다.

해당 클래스들은 사람이 직접 보더라도 문서 전체 레이아웃만으로는 구분이 어려우며,
문서 중앙 혹은 상단의 제목 텍스트를 확인해야만 구분이 가능한 수준의 높은 유사성을 가지고 있었습니다.
이로 인해 단순한 CNN 기반 전역 특징 학습만으로는 미세한 차이를 충분히 반영하기 어렵다는 한계가 드러났습니다.

### 시도 및 접근

이 문제를 해결하기 위해 다음과 같은 시도들을 진행하였습니다.

- 전처리 방식 조정
문서 이미지를 정사각형으로 강제 resize할 경우,
문서 상단 제목 영역이 잘려 핵심 정보가 손실될 수 있다고 판단하여
정사각형 crop 대신 종횡비를 유지하는 방식(letterbox 기반 resize)을 실험하였습니다.

- Title-focused Augmentation 실험
혼동이 심한 클래스(3/7, 4/14)를 대상으로
문서 상단 제목 영역에 주목하도록 유도하는 augmentation 전략을 적용하고,
혼동 행렬(confusion matrix) 및 오분류 이미지를 반복적으로 분석하였습니다.

- Soft Label 적용 시도
특정 샘플이 갖는 강한 hard label 신호가 오히려 모델을 혼동시킬 수 있다고 판단하여,
soft label을 부여함으로써 학습 시 특정 샘플의 영향력을 완화하는 전략도 실험하였습니다.

- 데이터 분리 전략에 대한 고민
반복적으로 오분류가 발생하는 샘플을
validation에서 제외하고 train에서만 활용하는 방식이 도움이 될 수 있을지에 대해서도 검토하였습니다.

### 한계 및 아쉬운 점

여러 전략을 설계하고 실험 방향을 구상하였으나,
프로젝트 후반부 일정 및 리소스 제약으로 인해
해당 접근법들을 최종 모델에 완전히 반영하지 못한 점이 가장 큰 아쉬움으로 남았습니다.

### 인사이트

이 경험을 통해 문서 분류 문제에서는
모델의 크기나 복잡도보다, 모델이 어떤 영역을 보고 판단하도록 유도하느냐가 훨씬 중요하다는 인사이트를 얻었습니다.
향후에는 클래스 간 혼동이 심한 경우, 구조적 모델 개선보다 관심 영역 기반 학습 전략을 우선적으로 고려할 필요가 있다고 판단하였습니다.

---

### 2. 2. Canny Edge Detection 기반 학습의 실용성 한계
### 한계

문서의 구조적 특징을 강조하기 위해
Canny Edge Detection 이미지를 학습 및 TTA에 활용하는 방안을 실험하였습니다.
일부 실험에서는 성능 개선 가능성이 관측되었으나,
전체 파이프라인에서 연산 비용과 학습 시간이 과도하게 증가하는 문제가 발생하였습니다.

### 결과 및 아쉬움

학습 및 추론 시간 병목으로 인해
해당 기법을 최종 파이프라인에 적용하지 못한 점이 아쉬움으로 남았습니다.

### 개선 방향

향후에는 edge 정보를 직접 입력으로 사용하기보다,

- edge를 보조 채널로 활용하거나
- 사전 처리된 feature 형태로 경량화하여 사용하는 방식
- 
등을 통해 성능과 효율 사이의 균형을 맞출 필요가 있을 것으로 판단됩니다.

---

### 3. Test 분포를 모사하기 위한 강한 전처리(Dirty Augmentation)의 한계
### 한계

로컬 검증 성능과 리더보드 성능 간 괴리를 줄이기 위해,
Test 데이터 분포를 최대한 모사하고자 ‘Dirty’ augmentation을 직접 구현하여 적용하였습니다.
이미지를 의도적으로 더럽히고 노이즈를 추가하는 방식으로
실제 Test 환경과 유사한 조건을 만들고자 하였습니다.

### 결과

로컬 환경에서는 초기 성능 하락이 관측되어
augmentation이 효과적으로 적용된 것으로 판단하였으나,
실제 리더보드 제출 결과에서는 성능 향상이 매우 제한적인 수준에 그쳤습니다.

### 아쉬운 점 및 개선 방향

해당 전략을 충분히 시각적으로 검증하지 못한 채 적용한 점이 아쉬움으로 남았습니다.
향후에는 단순히 augmentation 강도를 높이는 방식이 아니라,

- Test 데이터와의 차이를 정성적으로 비교·분석
- 실제 Test 이미지와 Train 이미지의 차이를 시각적으로 확인
- 단계적으로 augmentation을 적용하며 영향 분석

과 같은 접근을 병행한다면 더 효과적인 개선이 가능할 것으로 판단됩니다.

---

### 종합 인사이트 
- 문서 분류 문제는 데이터 이해와 문제 정의가 성능의 상한선을 결정하는 것 같습니다.
- 무작정 모델을 복잡하게 만들거나 augmentation을 강하게 주는 것이 항상 정답은 아니라고 생각이 들었습니다.
- 실제 서비스 및 대회 환경에서는 일반화 성능과 안정성이 무엇보다 중요하다는 생각이 들었습니다.
- 또한 현업에서의 비용 과 자원에 대한 고민을 하면서 프로젝트에 접근해야 하는 중요성을 깨닫게 되었습니다.


## 📍 회고
👑 이진성 : 저는 이번 딥러닝 프로젝트를 하면서 굉장히 많은 어려움을 느꼈습니다. 우선 머신러닝 모델링을 할 때에는 어떤 부분이 성능저하에 원인이 되는지 간접적으로 알 수 있는 요소들이 꽤 존재했습니다. (Feature Importance 를 찍어 보던지, 그리고 도메인 지식과 변수간의 상관관계가 어느정도 뚜렷한 부분들이 많았음.) 근데 딥러닝은 데이터마다 천차만별 인데다가 이게 정말 중요한 Augmentation 인지, 이게 왜 성능이 오르지? 왜 떨어지지? 를 판단할 수 있는 척도가 너무 부족하다는 느낌이 들어서 많이 삽질하고 힘들었던 프로젝트 였던 것 같습니다. (게다가 정확한 성능을 파악해볼 수 있는 리더보드 자체에 제출제한도 있어서 현업에서의 비용문제에 대한 고민을 같이 해볼 수 있는 좋은 기회였던 것 같습니다.) 또 기분탓인지는 모르겠지만 딥러닝 자체가 굉장히 추가적인 요소들에 민감하다는 생각을 많이 받았습니다. 소수 데이터가 성능 하락이나 향상에 영향을 준다던지.. 섬세하고 디테일 해야 좋은 성능을 보장할 수 있겠다 라는 생각이 들었고 무엇보다 데이터 품질이 프로젝트의 전반적인 성공여부를 결정할 수 있겠다 라는 인사이트도 얻게 되었습니다. 많이 어렵고 전체적인 팀 Score 에 많이 기여하지는 못했지만 그래도 여러 인사이트를 얻고 딥러닝 과 Pytorch 프레임워크에 문을 열 수 있는 좋은 기회였던 것 같습니다.

🙍 박세희 : 처음 경험해보는 end-to-end 딥러닝 프로젝트라 별거 아닌 곳에 하루를 다 보낸적도 있었고 제대로 된 결과물을 얻지 못한 하루가 더 많았지만 그렇게 직접 부딪혀 보며 많이 성장할 수 있었던 것 같습니다. 딥러닝을 해보며 느낀건 정확히 어떻게 해야 모델의 성능이 올라갈것이다 라고 정해진 것이 없으며 주어진 시간동안 최대한 이것저것 많은 시도를 효율적으로 해보는게 가장 중요한것 같다는 것입니다. 그러기 위해선 저만의 수많은 경험으로 쌓은 빅데이터가 정말 중요하겠다는 생각이 들었고 앞으로 그런 저만의 인사이트를 쌓기 위해 좀 더 노력해야 할 것 같습니다.

🙍 서효림 : 이번 프로젝트를 통해 단순히 모델 구조를 변경하는 것보다 데이터 특성 이해, 클래스 간 관계 분석, 데이터 분포 균형이 성능에 훨씬 큰 영향을 준다는 것을 체감할 수 있었습니다.
특히 팀원들의 코드를 공유받아 분석하는 과정에서, 제 코드가 과도한 규칙 기반 접근으로 일반화 성능을 해치고 있었음을 되돌아볼 수 있었습니다. 그 과정을 통해 단순한 구조와 안정적인 학습 전략이 오히려 더 강력할 수 있다는 인사이트를 얻었습니다.
혼자였다면 시도하지 못했을 다양한 접근과 실험을 팀 프로젝트를 통해 경험할 수 있었고, 성과와 관계없이 협업 과정 자체가 매우 값진 경험이었다고 느꼈습니다.

🙍‍♂ 유창준 : pretrained 모델로는 과적합 발생할 것 같아서 직접 모델링 도전했으나 Attention Memory 실험 도전, Augmentation 시도 못했던 아쉬움, triplet loss까지 시도 하지 못했던 아쉬움

🙍‍♂ 이건우 : 이번 딥러닝 프로젝트를 진행하면서 저는 여러 생각들을 하였습니다. 모델은 어떤모델로 하는게 좋을까? 좋은 데이터셋이란 무엇이고 어떤 데이터셋 증강을 가져가야하나 등등 여러 실험들과 실패들이 많았던 프로젝트임에도
저 개인적으로는 데이터의 특성(특히나 문서)에 집중한 전처리와 TTA 전략으로 성능을 꾸준히 우상향시키려고 노력한 프로젝트였고
데이터셋 선택 데이터 증강 전처리 tta 등 머신러닝에 대한 직관을 키운 의미가 있는 프로젝트였다고 생각합니다.


## 👨‍👩‍👧‍👧 협업
#### 🤝 협업일정 및 방식
- 협업일정 : 정규 수업시간 (09시 ~ 18시) 도 중 시간제약 없이 현재 진행상황 실시간 공유 및 18시 수업 종료 전 금일 진행했던 프로젝트 작업내용 공유
- 협업방식 : Slack
<img width="900" height="800" alt="Image" src="https://github.com/user-attachments/assets/81f398ac-8071-4cd6-b271-67ee838917be" />

- 미팅일정 : 각 개인의 프로젝트 진행율을 반영하여 회의 마지막 진행상황으로 다음 회의 일정 조율하기
- 미팅방식 : 실시간 Zoom 플랫폼을 통한 화면공유 및 실시간 음성 대화
  
![Image](https://github.com/user-attachments/assets/871e27df-0bc3-451c-b835-b0ffd373bdb8)

#### 📋 일정 및 프로젝트 관리 툴
- Notion
<img width="1318" height="935" alt="Image" src="https://github.com/user-attachments/assets/d48451e7-2344-4dff-86a5-1514af420237" /> <br>

- WAN DB
<img width="1185" height="666" alt="Image" src="https://github.com/user-attachments/assets/f2607638-71d2-45f1-a53e-2e86c58c7796" />

## 🌐 기술스택
[ConvNeXt] : https://docs.kanaries.net/ko/topics/ChatGPT/convnext

[Swin Transformer] : https://mr-waguwagu.tistory.com/32

[Conv vs Swin] : https://dsba.snu.ac.kr/seminar/?mod=document&uid=1912

[Augmentation] : https://velog.io/@xpelqpdj0422/3.-%EC%9E%98-%EB%A7%8C%EB%93%A0-Augmentation-%EC%9D%B4%EB%AF%B8%EC%A7%80-100%EC%9E%A5-%EC%95%88-%EB%B6%80%EB%9F%BD%EB%8B%A4
