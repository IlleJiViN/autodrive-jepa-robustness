# 실험 순서와 성과 기록 기준

1. 실제 BDD100K train/val 원본을 준비하고 manifest를 변환한다. 이미지/box/scene_id와 train/val 분리를 검사한다.
2. 200-step pilot으로 GPU 메모리, 유한 loss/gradient, target/context variance, 체크포인트 재개를 확인한다.
3. 동일 architecture의 `configs/phase1_clean.json` baseline과 `configs/phase1_vits.json` corruption-aware JEPA를 비교한다. train_families 외 학습 예산/seed/split과 평가 조건을 동일하게 유지한다.
4. seed17/23/29를 반복하고 representation robustness를 전 validation manifest에서 측정한다.
5. 동일 detection head와 학습 예산을 사용해 YOLO baseline/ViT adapter의 clean mAP와 corruption별 mAP를 비교한다.
6. object/block/temporal mask와 실제 adverse-weather holdout으로 확장한다.

clean config는 손상되지 않은 이미지의 visible patches로 masked clean target을 예측한다. clean과 corruption-aware checkpoint에 동일 네 corruption 평가를 적용한다.
baseline 준비와 실험 완료 전에는 corruption-aware 효과를 주장하지 않는다.

| 단계 | 현재 상태 | 결과 파일/지표 |
|---|---|---|
| 입력 데이터 | pilot train2,048/val256, legacy labels 연결 완료 | labeled manifest / labels_provenance.json |
| GPU pilot | 두 모델 각각200 step 완료 | paired_pilot_summary.json |
| ViT-S 추가 학습 | 같은 부분집합, seed17, 추가10 epoch 완료 | paired_continuation_summary.json / final_encoder.pt |
| 전체 데이터·여러 seed 학습 | 미실행 | 전체 manifest와 동일 예산의 seed17/23/29 반복 |
| 표현 robustness | selected/final 평가 완료 | cosine / identity retrieval / effective rank |
| 객체종류 존재 probe | frozen JEPA/clean/random 비교 완료 | class-presence macro AP,9 supported classes |
| 검출 평가 | 작은 subset custom probe 1-seed 완료; 낮은 절대 mAP | standardized YOLO baseline, multi-seed, full-data mAP / recall / calibration |
| 실제 악천후 일반화 | 후속 실험 | dataset/weather/severity holdout |
| Driving decision model | 연구 설계 단계 | task-specific labels / latency / safety metrics |

현재 검출 pilot은 frozen YOLOFeatureAdapter 뒤에 둔 custom three-scale probe head이며 Ultralytics YOLO가 아니다. 낮은 절대 mAP로 인해 standardized detector baseline, multi-seed, full-data 검증이 남았다. CLM/decision model을 이번 Phase 1 학습에 포함하지 않는다.
masked loss가 선택한 추가1-epoch encoder와 마지막10-epoch encoder를 구분한다. frozen probe는 마지막 모델로 고정했다.
부분집합의 손상 class-presence macro AP는 corruption0.429946, clean0.409994, random0.408860이다.
단일 seed/작은 val 결과이며 box detection mAP나 실제 악천후 generalization으로 해석하지 않는다.
학습 시간 추정과 기대 향상은 실측 성과와 구분한다. 각각의 결과에는 config, seed, split, 데이터/체크포인트 식별 정보와 평가 범위를 남긴다.
