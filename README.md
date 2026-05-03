# 👨‍🏫 Document Type Classification Competition
목적 : 여러 종류의 문서 이미지를 이미지 별로 문서 타입을 분류

## 👨‍👧‍👦 Team 
<table align="center">
  <!-- 상단 팀원 이름 Table-->
  <tr>
    <th align="center">👑 이진성</th>
    <th align="center">🙍 박세희</th>
    <th align="center">🙍 서효림</th>
    <th align="center">🙍‍♂ 유창준</th>
    <th align="center">🙍‍♂ 이건우</th>
  </tr>
  
  <!-- 팀원 이미지 Table-->
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/a9befe05-0cb9-4e0c-ba1f-43b4275ce55e" width="130">    <!--이진성-->
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8640b497-ee8b-426f-b314-e6681ac5e0a6" width="130">    <!--박세희-->
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/5b02c89e-5860-43a6-9adc-d1e2944952f6" width="115">    <!--서효림-->
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/11d4493f-10d3-4342-8bc8-3c78353626e1" width="130">    <!--유창준-->
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/11d4493f-10d3-4342-8bc8-3c78353626e1" width="130">    <!--이건우-->
    </td>
  </tr>

  <!-- 팀원 역할 Table-->
  <tr>
    <!--이진성-->
    <td align="center">
      EDA <br>
      Data Preprocessor <br>
      Modeling
    </td>
    <!--박세희-->
    <td align="center">
      EDA <br>
      Data Preprocessor <br>
      Modeling
    </td>
    <!--서효림-->
    <td align="center">
      EDA <br>
      Data Preprocessor <br>
      Modeling
    </td>
    <!--유창준-->
    <td align="center">
      EDA <br>
      Data Preprocessor <br>
      Modeling
    </td>
    <!--이건우-->
    <td align="center">
      EDA <br>
      Data Preprocessor <br>
      Modeling
    </td>
  </tr>

  <!-- GitHub URL-->
  <tr>
    <td align="center">
      <a href="https://github.com/wlstjd6524"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>        <!--이진성-->
    </td>
    <td align="center">
      <a href="https://github.com/carolinespwork"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>    <!--박세희-->
    </td>
    <td align="center">
      <a href="https://github.com/S202010741"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>        <!--서효림-->
    </td>
    <td align="center">
      <a href="https://github.com/NullXeronier"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>      <!--유창준-->
    </td>
    <td align="center">
      <a href="https://github.com/lgw2000"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/></a>           <!--이건우-->
    </td>
  </tr>

</table>


## 📺 Presentation
[발표자료](https://github.com/user-attachments/files/25160421/3.Document.Classification.pdf)

## 📂 ReadME Index 
[🎯 Project Overview (프로젝트 개요 및 목표)](#project-overview) <br>

[⏱️ Project Duration & 🔧 Tech Stack (기간 및 기술스택)](#projectduration-techstack) <br>

[📊 Data Analysis & Hypothesis (데이터 분석 및 실험 방향성 설정)](#data-analysis) <br>

[🚀 Experimental Progression (실험 과정 및 빌드업)](#experimental-progression) <br>

[🧪 Final SOTA Architecture & Result (핵심 실험과 최종 아키텍처 및 최종결과)](#final-sota-architecture) <br>

[🛠️ Troubleshooting & Engineering (문제 해결 및 인프라 안정화)](#troubleshooting-engineering) <br>

[👥 Team Leadership & Management (팀 리더십 및 협업)](#teamleadership-management) <br>

[📈 Retrospective & Future Work (회고 및 향후계획)](#retrospective-futurework)


<a id="project-overview"></a>

## 🎯 Project Overview
### 프로젝트 배경
- 실무 현장에서 수집되는 아날로그 문서 데이터(금융, 의료, 보험, 물류 등) 의 디지털화 니즈가 증가하는 추세입니다.
- 이에 따라 다양한 양식으로 혼재된 아날로그 문서 데이터의 종류를 자동으로 식별하고 분류하는 시스템 구축이 요구됩니다.

### 핵심 과제
- 절대적 데이터 볼륨 부족 극복 : 17개의 복잡한 문서 클래스를 분류해야 함에도 불구하고 전체 학습 데이터는 1,570장에 불과하여 발생하는 과적합 위험을 방지할 강력한 일반화 및 데이터 증강 전략을 수립.
- 미공개 데이터 노이즈 대응 및 강건성 확보 : 구겨짐, 빛 번짐 등 실제 현실의 노이즈가 섞인 3,140장의 테스트 데이터에 대응하기 위해, EDA를 통한 노이즈 유추 및 모델의 강건성 향상.
- 형태적, 의미적 유사성이 높은 클래스 식별력 강화 : 전체 레이아웃이 거의 동일하고 단 한글자 차이(ex : 입퇴원확인서 vs 진료확인서, 진단서 vs 소견서)로 분류가 갈리는 문서들을 정확히 구분하기 위해 타겟팅 기반의 특징 추출 전략 설계.
- 클래스 간 데이터 불균형 해소 : 클래스별 이미지 수가 46장에서 100장 사이로 랜덤하게 분포된 불균형 환경을 해결하기 위한 검증(K-Fold) 전략 및 가중치 최적화 방향 수립.

### 핵심 평가 지표 및 최적화 목표
단순한 직관전 모델 튜닝을 지양하고, 철저한 EDA 를 기반으로 한 '데이터 중심' 의 문제 해결을 달성하기 위해 다음 지표와 목표를 핵심 타겟으로 삼았습니다.
  - Macro F1-Score : 데이터의 불균형이 존재하는 환경을 고려하여, 다수 클래스에 편향되지 않고 소수 클래스의 예측 성능까지 종합적으로 평가할 수 있도록 최종 평가지표 `F1 Score` 로 설정.
  - 데이터 품질 최우선 최적화 : 모델의 복잡도를 무작정 높이기보다, 데이터의 특성(해상도, 종횡비 편차 등)을 이해하고 노이즈에 강건한 전처리 파이프라인을 구축하는 것을 최우선 목표로 설정.
  - 일반화 성능 극대화 : 로컬 검증 성능에 매몰되지 않고, 실제 리더보드에서의 점수 괴리를 줄이기 위한 객관적 평가 체계 확립.


<a id="projectduration-techstack"></a>

## ⏱️ Project Duration & 🔧 Tech Stack
### ⏱️ Project Duration
- 2026.01.23 ~ 2026.02.05

### 🔧 Tech Stack
| Category | Tech Stack |
| :--- | :--- |
| **Language** | Python 3.10.0 |
| **Hardware** | Single GPU (NVIDIA RTX 3090, 24GB VRAM) |
| **Deep Learning Framework** | PyTorch |
| **CV Models (Backbone)**| ResNet-50 (Baseline), ConvNeXt, Swin Transformer |
| **Model Library** | timm (PyTorch Image Models) |
| **Data Augmentation & Image Processing** | Albumentations, OpenCV (cv2) |
| **Machine Learning Utils** | Scikit-learn (StratifiedKFold) |
| **Data Analysis & EDA** | Pandas, NumPy |
| **Experiment Tracking & Metric** | Weights & Biases (wandb), F1 Score |
| **Collaboration** | Slack, Zoom |


<a id="data-analysis"></a>

## 📊 Data Analysis & Hypothesis
### 📷 기본 데이터 구성
- 학습 데이터  : 총 1570장의 문서 이미지가 존재하며 17개 클래스 안에 각 클래스 별로 46 ~ 100 장 이미지가 랜덤으로 분포되어 있습니다.
- 테스트 데이터 : 총 3140장의 문서 이미지가 존재하며 여러 Augmentations 이 적용되어있습니다. 어떤 Augmentations 는 Secret Data 로 존재합니다.

### 🧾 Class 정보
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

### Insight 1. 클래스 분포 불균형을 고려한 검증 및 평가 가설
<img width="1212" height="547" alt="Image" src="https://github.com/user-attachments/assets/9cd06cbc-c937-4983-ad6f-3fcb8822a726" /> <br>

- 분석 : Train 데이터의 클래스 분포를 시각화한 결과, 대다수의 클래스는 100 장의 이미지를 가지나 일부 클래스는 46~50장 수준으로 절반에 불과한 불균형이 확인되었습니다.
- 실험방향 : 단순 Accuracy 를 평가지표로 사용할 경우, 다수 클래스에 편향된 결과를 낳을 위험이 매우 높다고 판단했습니다.
    - 해결 및 가설 : 모델이 소수 클래스의 특성도 놓치지 않도록 평가지표를 F1-Score 로 설정하려고 했습니다. 또한, 검증 셋 분리시 일반적인 K-Fold 가 아닌 클래스 비율을 유지하는 Stratified K-Fold 가 도입되면 좋겠다 라는 가설을 세웠고, 학습 손실함수 설계시 클래스 가중치를 부여하는 방향으로 실험을 전개해야 한다는 가설을 수립했습니다.
 
---

### Insight 2. Train / Test 데이터 간 종횡비 불일치
<img width="992" height="574" alt="Image" src="https://github.com/user-attachments/assets/0ad89ac4-0ed0-40c1-8788-f19916be0e35" /> <br>

- 분석 : Train 과 Test 데이터의 Aspect Ratio 분포를 겹쳐본 결과, 극명한 차이가 발견되었습니다. Train 샘플은 0.75 비율(세로가 긴 형태)에 극단적으로 밀집되어 있는 반면, Test 샘플은 1.25 ~ 1.4 주변의 비율(가로가 긴 형태) 에서도 높은 밀도를 보였습니다.
- 실험방향 : 문서 이미지 분류에서 레이아웃은 핵심 특징입니다. 기존의 일반적인 이미지 분류처럼 224 * 224 등의 정사각형으로 강제 리사이징 할 경우, 텍스트 비율이 찌그러지거나 중요한 형태적 특징(ex : 입원 및 퇴원 확인서나 소견서 같은 부분들은 레이아웃도 비슷한데다가 안에 내용들도 어느정도 의학 정보가 담겨 있기 때문에 일반 사용자가 보기에도 해당 데이터 샘플들은 위의 목차를 통해 문서 특징을 분간 할 수 있음.)이 손실될 수 있다고 생각했습니다.
    - 해결 및 가설 : 일반적인 Resize 연산을 폐기하고, 원본 종횡비를 유지하면서 빈 공간을 패딩으로 채우는 Letterbox Resize 방식을 도입해야 한다는 방향성을 설정했습니다. 또한, 고정된 해상도 입력에 강건한 모델 아키텍처 선정이 필요하다는 가설을 세웠습니다.

---

### Insight 3. 밝기(Brightness) 분포 차이를 통한 Secret 노이즈 유추
<img width="1389" height="590" alt="Image" src="https://github.com/user-attachments/assets/6162c5cb-3170-4efa-bf69-4fbfde690d59" /> <br>

- 분석 : Train 이미지의 밝기는 100 ~ 175 사이에 다소 안정적으로 분포되어 있습니다. 반면, Test 데이터는 25 ~ 125 구간(매우 어두움)에 큰 피크가 존재하며 전체적으로 노이즈가 강하게 낀 것을 간접적으로 확인했습니다. 실제 Test 데이터 시각화 시에도 빛 번짐, 어두운 환경에서 촬영된 문서들이 다수 관측되었습니다.
- 실험방향 : Train 데이터 원본만으로 학습시킬 경우, Test 환경의 노이즈를 견디지 못하고 리더보드 점수가 하락하는 '일반화 실패' 가 발생할 것이 자명했습니다.
    - 해결 및 가설 : Test 데이터의 분포를 모사하기 위해 픽셀 단위의 강한 노이즈 증강 파이프라인을 구축해야 한다는 가설을 설정했습니다.


<a id="experimental-progression"></a>

## 🚀 Experimental Progression
### Phase 1. Baseline 구축 및 데이터 불균형 방어
- 가설 및 실험 : 문서의 지역적 특징 추출에 강점이 있는 `ConvNext-Base` 를 베이스 라인 모델로 선정했습니다. 극심한 클래스 불균형 문제를 방어하기 위해 StratifiedKFold(n_splits=5) 를 적용하여 검증 셋을 구성하고, 학습 손실 함수에 Soft Target Cross Entropy 를 도입했습니다.
- 한계직면 : 로컬 검증에서는 F1 Score 가 0.95 이상의 높은 점수를 달성했으나, 리더보드 제출 시 심각한 성능 하락을 경험했습니다.

---

### Phase 2. Data-Centric 최적화 : Label Nosie 탐지 및 Soft Labeling
- 가설 및 실험 : 성능 하락의 원인이 학습 데이터 내의 잘못된 정답지에 있다고 판단했습니다. 5-Fold OOF 예측값을 바탕으로 Confusion Matrix 를 분석한 결과, 외관이 매우 유사한 입퇴원 확인서(Class 3), 외래진료증명서(Class 7), 소견서(Class 14) 등 특정 클래스에서 오분류가 집중됨을 확인했습니다.
- 해결책 : 예측 확신도가 0.90 이상이면서 정답이 틀린 33개의 샘플을 '의심군' 으로 추출했습니다. 모델이 잘못된 하드 레이블에 과적합되는 것을 막기 위해, 이 추출된 의심군 데이터들에 한해 1.0 의 하드 정답이 아닌 OOF 확률값을 `soft_targets_prob` 로 변환하여 주입하는 파이프라인을 구축했습니다.

---

### Phase 3. 종횡비(Aspect Ratio) 보존을 위한 자체 Sampler 구현
- 가설 및 실험 : EDA 에서 파악한 'Tran / Test 데이터 간의 극단적인 종횡비 차이'를 해결해야 했습니다. 일반적인 224 * 224 강제 리사이징은 문서의 핵심인 텍스트와 레이아웃 비율을 심각하게 왜곡시켰습니다.
- 해결책 : 원본 비율을 유지하면서 빈 공간을 패딩하는 `use_letterbox = True` 방식을 도입했습니다. 더 나아가 학습 효율을 극대화하기 위해, 비슷한 종횡비를 가진 이미지들끼리 동적으로 배치를 묶어주는 `BucketBatchSampler` 를 자체 구현하여 적용했습니다.

---

### Phase 4. 로컬 검증(Validation) 신뢰성 회복 - Dirty Holdout 구축
- 문제 정의 : 로컬 검증 점수와 실제 리더보드 점수 간의 극심한 격차를 겪으며, 기존의 랜덤 기반(K-Fold) 검증 방식이 테스트 데이터의 실제 분포를 대변하지 못한다는 치명적인 한계를 인지했습니다.
- 가설 및 실험 : 로컬 평가 지표의 신뢰성을 회복하기 위해 검증 셋 파이프라인을 전면 수정했습니다. Train Data 내에서 노이즈 수준(빛 번짐, 구겨짐, 대비 등)이 높은 데이터들을 수학적으로 선별하여, 실제 테스트 환경과 유사한 악조건을 가진 `Dirty Holdout` 검증 셋을 자체 구축했습니다.
- 평가 고도화 : 이에 더해 추론 단계에서도 스캔 과정의 미세한 틀어짐을 보정하기 위해 -15° ~ +15° 사이의 미세 각도 변환을 포함하는 `Adaptive TTA` 를 구현하여, 단순 과적합이 아닌 실제 노이즈 환경에서의 강건성을 객관적으로 평가할 수 있도록 개선했습니다.

---

### Phase 5. 단일 모델의 한계 인식과 SOTA 앙상블로의 피벗
위와 같은 여러 Phase 들을 나누면서 치열한 데이터 파이프라인 최적화(`Bucket Sampler`, `Soft Labeling`, `Dirty Holdout`)를 통해 로컬과 리더보드 간의 격차를 크게 좁히는 데 성공했습니다.

- 빌드업 : 여러 시드값에 따른 가중치 앙상블 테스트를 진행한 결과, 단일 구조 모델의 최적화만으로는 미세한 형태 차이를 완벽히 분류하는 데 한계가 있음을 객관적 데이터로 확인했습니다. 이를 돌파하기 위해, ConvNext 와 더불어 문서 전체의 전역 문맥을 파악하는데 특화된 Swin Transform 모델을 도입하여 이중 아키텍처 앙상블을 구축하는 최종 SOTA 파이프라인으로 선회하였습니다. 


<a id="final-sota-architecture"></a>

## 🧪 Final SOTA Architecture & Result
여러가지 파이프라인을 진행해보면서 겪었뎐 병목현상을 극복하기 위해, 서로 다른 특성을 가진 두 개의 강력한 백본(CNN + Transformer) 을 병합하는 이중 아키텍처 앙상블을 팀의 최종 SOTA Model 파이프라인으로 채택했습니다.
단순한 하이퍼 파라미터 튜닝을 넘어 데이터 정합성 보정, 문서 특화 전처리, 그리고 Logit 단위의 가중합 앙상블을 통해 일반화 성능을 극대화 했습니다.

<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/b51f24ca-7954-44a9-9442-b0abc99969db" /> <br>

### 1. Data-Centric 최적화 및 정합성 보정
- Label Correction (오라벨 교정) : `train.csv` 내 7개의 치명적인 잘못된 라벨을 사전 분석하여 수동 교정했습니다. 이를 파이프라인 내 `correct_labels` 로직으로 자동화하여, 모든 모델이 100% 동일하게 정제된 데이터를 학습하도록 일관성을 확보했습니다.
- 문서 특화 고급 전처리
  - `Deskew` : 이미지 픽셀 구조를 분석하여 스캔 과정에서 발생한 문서의 미세한 기울기를 자동으로 섬세하게 보정했습니다.
  - `Letterbox Resize` & `Aspect Ratio Bucketing` : 강제 리사이징으로 인한 문서 왜곡을 막기 위해 원본 종횡비를 유지하며 빈 공간을 패딩하고, 비슷한 비율의 이미지끼리 동적으로 배치를 묶어내는 버킷팅 기법을 구현했습니다.
- Mixup & Label Smoothing : 두 모델 모두 `timm` 라이브러리의 Mixup 기법과 `SoftTargetCrossEntropy` 를 도입했습니다. 극단적인 하드 라벨(0, 1) 대신 Label Smoothing 이 적용된 타겟을 활용하여 모델의 과적합을 방지했습니다.

### 2. SOTA 아키텍처 구성 및 학습 파이프라인
- 이중 아키텍처 전략
  - ConvNext-Base : CNN 계열 특유의 강력한 지역 패턴 학습 능력을 활용해 문서 이미지의 텍스트와 레이아웃 디테일을 추출했습니다.
  - Swin-Base : 윈도우 기반 Self-Attention 을 통해 문서의 전역 문맥을 파악하며, 384 * 384 의 고정 해상도에서 안정적인 학습을 수행했습니다.
- 학습 및 최적화 설정
  - 검증 체계 : 5-Fold Stratified Cross Validation 을 기반으로, 클래스 불균형 문제를 상쇄하기 위해 Class Weight 가 반영된 Validation Loss 를 평가지표로 활용했습니다.
  - 학습 안정화 : 메모리 최적화를 위한 AMP(Mixed Precision) 와 `CosineAnnealingWarmRestarts` 스케줄러를 적용했습니다.
  - Early Stopping : 각 모델의 학습 속도에 맞춰 ConvNext 는 Patience 5, Swin 은 Patience 7 을 적용해서 각 Fold 별 최적의 CheckPoint 를 디스크에 저장했습니다.
 
### 3. 추론 최적화(TTA) 및 Final Logit Ensemble
- Fold Ensemble & TTA : 테스트 추론시 0°, 90°, 180°, 270°의 `rot90` TTA 를 부여한 뒤, 각 Fold 에서 도출된 Logit 값들의 평균을 내어 모델별 1차 안정성을 극대화했습니다.
- 가중합 기반 최종 앙상블
  - 각 모델이 예측한 최종 확률값을 `predict_logits.pt` 로 저장하고 분리하여 메모리 병목을 없앴습니다.
  - Softmax 통과 이전의 Logit 단계에서 ConvNext 와 Swin 의 비율을 조절해가며 앙상블 가중합을 수행했습니다.
  - 실험 결과, 앙상블 가중치를 여러 비율을 섞으면서 테스트 Submission 을 해보면서 최적의 비율인 ConvNext : Swin = 0.7 : 0.3 의 가중치로 병합한 후 `argmax` 를 취했을 때 리더보드 환경에서 가장 강력하고 안정적인 최종 성능이 도출됨을 증명했습니다.
 
### 🏆 Final Performance & Leaderboard
- Final Model : ConvNext & Swin 0.7:0.3 Ensemble
- F1 Score : 0.9174

<img width="980" height="376" alt="Image" src="https://github.com/user-attachments/assets/298c49e6-93c4-4eac-a171-099a68f3a1fb" /> <br>
Rank: 🥉3rd


<a id="troubleshooting-engineering"></a>

## 🛠️ Troubleshooting & Engineering
### 1. 제한된 하드웨어 리소스 극복 및 메모리 안정화
#### 문제 정의
파라미터가 큰 Swin-Base 와 ConvNext 모델을 고해상도로 학습시키는 과정에서 24GB VRAM 환경의 단일 GPU 메모리 한계에 빈번하게 부딪혀 학습 파이프라인이 중단되는 문제가 발생했습니다.

#### 원인 분석
모델 자체의 크기뿐만 아니라, 5-Fold 교차 검증을 진행하며 각 Fold 가 넘어갈 때 마다 이전 Fold 의 모델 객체와 데이터 로더가 GPU 메모리에서 완전히 해제되지 않아 발생하는 메모리 누수 및 단편화가 원인이었습니다.

#### 해결 방안
- Gradient Accumulation : 물리적인 Batch Size 를 강제로 줄이는 대신 `accmulation_stpes = 4` 를 적용하면서, OOM(Out Of Memory) 을 방어하면서도 실질적인 Batch Size 를 유지했습니다.
- 명시적 캐시 클리어 : 각 Fold 종료 시그널에 맞춰 `gc.collect()` , `torch.cuda.empty_cache()` 를 강제 호출하는 `cleanup_cuda()` 파이프라인을 구축했습니다.
- 메모리 파편화 제어 : 환경 변수 `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"` 을 설정하여 PyTorch 의 캐시 할당자 동작을 최적화했습니다.

#### 인사이트
딥러닝 모델링은 단순히 수학적 정확도를 높이는 것을 넘어, 주어진 하드웨어 인프라 자원을 얼마나 극한으로 효울화하여 사용할 수 있는가에 대한 '엔지니어링 최적화'가 동반되어야 함을 체감했습니다.

---

### 2. 앙상블 파이프라인 병목 현상 및 시간 비용 최적화
#### 문제 정의
TTA 가 적용된 여러 Fold 의 모델들을 추론 시점에 동시에 메모리에 올려 앙상블을 진행하려고 하자, 극심한 시간 지연과 추론 서버의 OOM 이 발생했습니다.

#### 원인 분석
최적의 앙상블 가중치를 찾기 위해 비율을 변경할 때마다 무거운 모델의 예측 함수를 처음부터 다시 태우는 비효율적인 구조로 코드가 설계되어 있었습니다.

#### 해결 방안
추론 단계와 앙상블 단계를 물리적으로 분리했습니다.
  - 무거운 모델 추론은 단 한번만 실행하여 산출된 예측 확률을 `predict_logits.pt` 파일로 디스크에 직렬화 했습니다.
  - 이후 별도의 가벼운 앙상블 스크립트를 구축하여, 캐싱된 Logit 파일만 로드한 뒤 CPU 환경에서 ConvNext 와 Swin 의 가중치를 0.75 부터 1.00 까지 0.05 단위로 즉각적으로 Sweep 하며 최적의 비율을 탐색하도록 자동화 했습니다.

#### 인사이트
반복되는 병목 구간을 캐싱을 통해 분리함으로써, 실험에 소요되는 시간 비용을 혁신적으로 단축할 수 있었습니다. 현업에서의 배포 환경과 서비스 추론 속도를 고려한 아키텍처 설계의 중요성을 깨달았습니다.

---

### 3. 실용성과 성능의 트레이드오프(Trade-off) 조율
#### 문제 정의
문서 클래스 간 유사성이 높아(예: 진단서 vs 소견서) 모델의 형태적 식별력을 끌어올리기 위한 추가적인 전처리가 필요했습니다.

#### 원인 분석
초기에는 원본 이미지의 외곽선 정보를 극대화할 수 있는 Canny Edge Detection 이미지를 생성하여 입력 채널에 추가하거나 TTA에 활용하면 레이아웃 식별 성능이 개선될 것으로 보았습니다.

#### 해결 방안
해당 방식을 로컬 환경에서 테스트해 본 결과, 성능의 미세한 향상 가능성은 보였으나 전처리 파이프라인의 연산 비용과 추론 시간이 기존 대비 기하급수적으로 증가하는 것을 확인했습니다. 팀 내 논의를 거쳐, 자원 소모 대비 이점이 적다고 판단하여 최종 SOTA 모델에서는 과감히 폐기하고 가벼운 Letterbox Resize로 대체했습니다.

#### 인사이트
단순히 모델의 Score를 0.01 올리는 것보다, 연산 비용과 추론 시간 증가율을 계산하여 "이 기술이 정말 상용화 가치가 있는가?"를 묻는 실용성(속도/자원) 중심의 밸런스 조율이 필수적임을 배웠습니다.


<a id="teamleadership-management"></a>

## 👥 Team Leadership & Management
단순 코드 병합 중심의 개발을 넘어, '실험 결과와 인사이트의 병합' 이 핵심인 프로젝트 특성에 맞춰 다음과 같은 엄격한 연구 협업 룰을 세팅하고 리드했습니다.

1. 1일 1회 리더보드 제출 : 완벽주의에 빠져서 로컬 환경에서 실험만 반복하는 것을 방지하고자 했습니다. 매일 정략적 지표를 강제로 확인하게 하여 파이프라인의 방향성을 끊임없이 수정할 수 있게 리드했습니다.
2. 데이터 기반 소통 규칙 : 추상적인 텍스트 공유를 금지했습니다. EDA 결과나 실험 인사이트를 Slack 에 공유할 때는 반드시 시각화된 데이터나 통계 지표를 첨부하도록 규정하여 팀 내 커뮤니케이션의 객관성을 확보하고자 했습니다.
3. 데일리 랩업 : 매일 정규 시간 전, 각자가 개별적으로 진행한 파이프라인 실험 결과와 성능 향상 로직을 투명하게 공유했습니다. 이를 통해 성공적인 실험 요소들이 팀원 전체의 파이프라인에 즉각적으로 이식될 수 있는 선순환 구조를 만들고자 했습니다.
4. W&B 기반 실험 트래킹 : 개인 로컬 환경에서 발생하는 실험 파편화를 막기 위해 Weight & Biases 를 연동하여 프로젝트 관리를 지시했습니다. 모든 실험의 하이퍼파라미터와 평가지표를 실시간으로 중앙 집중화하여, 주관적 판단이 아닌 Metric 기반의 판단 시스템을 정착시키고자 했습니다.

<img width="1185" height="666" alt="Image" src="https://github.com/user-attachments/assets/36a1a970-1fab-4f1e-ac9f-9bc05386cd33" /> <br>
협업 플랫폼 : Slack, Zoom


<a id="retrospective-futurework"></a>

## 📈 Retrospective & Future Work
### 📍 회고
저는 이번 딥러닝 프로젝트를 하면서 굉장히 많은 어려움을 느꼈습니다. 우선 머신러닝 모델링을 할 때에는 어떤 부분이 성능저하에 원인이 되는지 간접적으로 알 수 있는 요소들이 꽤 존재했습니다. (Feature Importance 를 찍어 보던지, 그리고 도메인 지식과 변수간의 상관관계가 어느정도 뚜렷한 부분들이 많았음.) 근데 딥러닝은 데이터마다 특징이 천차만별 인데다가 이게 정말 중요한 Augmentation 인지, 이게 왜 성능이 오르지? 왜 떨어지지? 를 판단할 수 있는 척도가 너무 부족하다는 느낌이 들어서 많이 삽질하고 힘들었던 프로젝트 였던 것 같습니다. (딥러닝의 고질적 문제인 블랙박스 형태라는 부분을 깨달았고, 게다가 정확한 성능을 파악해볼 수 있는 리더보드 자체에 제출제한도 있어서 현업에서의 비용문제에 대한 고민을 같이 해볼 수 있는 좋은 기회였던 것 같습니다.) 또 기분탓인지는 모르겠지만 딥러닝 자체가 굉장히 추가적인 요소들에 민감하다는 생각을 많이 받았습니다. 소수 데이터가 성능 하락이나 향상에 영향을 준다던지.. 섬세하고 디테일 해야 좋은 성능을 보장할 수 있겠다 라는 생각이 들었고 무엇보다 데이터 품질이 프로젝트의 전반적인 성공여부를 결정할 수 있겠다 라는 인사이트도 얻게 되었습니다. 또한 딥러닝 과 Pytorch 프레임워크에 문을 열 수 있는 좋은 기회였던 것 같습니다.


### 📗 향후계획
프로젝트 진행 과정에서 Class 3, 7, 14 간의 오분류가 반복적으로 관측되었던 부분이 있었습니다. 이 부분은 사람조차 전체 레이아웃만으로 구분이 힘든 일부 문서들이 존재하던 Class 였고 이 Class 들을 다루면서, 모델의 크기보다는 '어디를 보게 할 것인가' 를 유도하는 전략이 훨씬 중요함을 배웠습니다. 이를 해결하기 위해 종횡비를 유지하는 Letterbox Resize 나 Title-focused 증강, Soft Labeling 등 다양한 가설을 세우고 실험을 했습니다. 비록 프로젝트 후반부의 시간과 리소스 제약으로 이 모든 가설을 최종 파이프라인에 녹여내진 못했지만, 향후 유사한 문제를 만난다면 무의미한 모델 구조 변경보다 <b>관심 영역 기반의 학습 전략</b> 을 1순위로 설계해보려고 합니다.

또한, 성능 향상과 <b>현업의 실용성 사이의 트레이드오프</b> 에 대해서도 깊이 고민하게 되었습니다. 문서의 구조적 특징을 강조하기 위해 Canny Edge Detection 을 도입했을 때, 성능 개선 가능성은 보였으나 학습 추론 병목 현상이 심하게 발생했던 케이스가 있었습니다. 이 실패를 거울삼아, 앞으로는 Edge 정보를 무거운 직접 입력으로 쓰기보다 보조 채널이 사전 처리된 Feature 형태로 경량화하여 자원 한계에 부합하는 효율적인 아키텍처를 구성하려 합니다.

가장 큰 아쉬움이자 교훈은 로컬과 실제 테스트 환경의 분포 격차를 줄이기 위해 시도했던 <b>Dirty Augmentation 실험</b> 이였습니다. 테스트 셋의 악조건을 모사하고자 이미지를 의도적으로 훼손해 보았지만, 결과적으로 리더보드 점수 향상은 미미했습니다. 이를 통해 증강 기법을 도입할 때는 단순히 노이즈 강도만 높이는 맹목적 접근을 지양하고, 실제 데이터 간의 정성적인 시각적 비교와 단계별 영향도 분석이 반드시 병행되어야 함을 깨달았습니다.

결과적으로 단순한 정확도 맞추기를 넘어서, 실제 서비스 환경에서 요구되는 <b>일반화 성능과 시스템적 안정성, 그리고 리소스 비용</b> 까지 종합적으로 고려하는 현업에서의 고민거리를 몸소 체험할 수 있었던 것 같습니다.
다음에는 이런 배웠던 부분을 통해 Computer Vision 과 관련된 프로젝트를 진행할 때 발판삼아 조금 더 좋은 성능과 조금 더 현업스럽고, 현업 다운 고민을 점진적으로 생각하면서 프로젝트에 접근해보고 싶습니다.
