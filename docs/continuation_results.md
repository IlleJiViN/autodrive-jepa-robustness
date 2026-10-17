# JEPA 추가10-epoch 학습과 라벨 기반 표현 평가

2026-10-06, Coder `owner/jwj-01`, `/home/coder/autodrive-jepa-robustness`, branch `codex/phase1-start`.
이전 [200-step pilot](pilot_results.md)을 이어 진행했다. 이미지 범위는 train2,048장/val256장이다.

## 완료한 작업

- Corruption/clean JEPA의 full `last.pt`에서 각각 추가10 epoch,2,560 optimizer steps 학습했다.
- 초기200 steps 포함 마지막 모델은 각각 누적2,760 steps다. 새20,480 image presentations,기존 포함22,080회다.
- `--warm-start`는 모델 상태를 이어받고 optimizer/RNG/epoch/LR 스케줄을 새로 시작한다. 정확한 `--resume`과 구분했다.
- 같은 architecture/데이터 순서/seed17/batch8를 사용했다. 추가 LR5e-5,EMA0.996→1,BF16이다.
- legacy detection annotation을2,304장 전부에 연결하고 YOLO export를 준비했다.
- selected/final encoder 표현 평가와 마지막 encoder의 frozen class-presence probe를 실행했다. 무학습 동일 ViT도 비교했다.

## 라벨과 평가한 체크포인트

| 연결한 split | 이미지 | 유효 박스 |
|---|---:|---:|
| train | 2,048 | 37,868 |
| val | 256 | 4,978 |

annotation basename/split,이미지 SHA-256/헤더,box 범위,train/val scene/content 분리를 확인했다.
원본은 legacy image별 `frames.objects` 형식이다. Detection2020 annotation 또는 전체80,000장 실험으로 표기하지 않는다.
기존 image-only manifest를 보존하고 새 `_labeled.jsonl`을 만들었다. 출처·alias·제외 규칙은 [data_source.md](data_source.md)에 있다.

두 모델 모두 masked validation loss의 최솟값은 추가 첫 epoch(step256)에서 나왔다.
`encoder.pt`/`best.pt`는 누적456-step 모델이다. `last.pt`/`final_encoder.pt`는 누적2,760-step 모델이다.
아래 주 결과와 선형 probe에는 마지막 모델을 사용했다.

## 마지막 모델의 표현 평가

val256장 전체,네 synthetic 손상 × severity1–5,조건별256쌍이다.
identity top1은 손상 이미지로256개의 clean 이미지 중 원래 이미지를 찾는 비율이다. 무작위 확률은0.390625%다.
객체 class 정확도나 box detection 지표가 아니다.

| 손상 | Corruption cosine | Clean cosine | Corruption identity top1 | Clean identity top1 |
|---|---:|---:|---:|---:|
| 안개 | 0.901478 | 0.287916 | 34.84% | 2.19% |
| 비 | 0.999818 | 0.998817 | 100.00% | 90.70% |
| 모션 블러 | 0.999391 | 0.999870 | 98.52% | 99.92% |
| 저조도 | 0.859971 | 0.666917 | 16.64% | 1.25% |
| 20조건 평균 | 0.940164 | 0.738380 | 62.50% | 48.52% |

cosine 차이는+0.201784다. 모션 블러 identity top1은 clean 모델이 높았고 손상별 효과가 달랐다.
표준편차는0.573637/0.480011로 기존0.001 기준의 collapse 경고는 발생하지 않았다.
그러나 centered effective rank는2.60/4.12로 낮아 표현 변동이 소수 방향에 집중되어 있었다.
높은 cosine만으로 의미 표현이나 운전 성능 개선을 주장하지 않는다.

selected encoder의20조건 평균 cosine은0.908335/0.719135,identity top1은52.17%/45.51%였다.
selected/final 각각의 전체 수치는 `runs/continuation_diagnostics_summary.json`에 있다.

## 실제 라벨의 frozen class-presence probe

box category를 이미지별 binary multi-label target으로 바꿨다. encoder를 고정하고 pooled384-d feature에 선형10-class head를 학습했다.
표준화는 train에서만 계산했다. 모든 encoder에 같은 seed17,50 epochs,batch128,AdamW LR0.001/weight decay0.01,
class-balanced BCE를 적용했다. head는 validation 선택 없이 마지막 epoch로 고정했다.
JEPA/clean은 마지막 누적2,760-step 모델,random은 같은 초기 seed/architecture의 무학습 모델이다.

val에 기차(train) class 양성이 없어 AP를 계산 가능한9개 class를 같은 방식으로 평균했다.
이미지 내 객체종류 존재 순위를 평가하는 AP이며 box 위치/IoU의 detection mAP와 다르다.
계산에는 [scikit-learn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)를 사용했다.

| Frozen encoder | Clean class-presence macro AP | 손상20조건 평균 macro AP |
|---|---:|---:|
| Corruption JEPA | 0.447233 | 0.429946 |
| Clean JEPA | 0.430532 | 0.409994 |
| 무학습 ViT | 0.428946 | 0.408860 |

| 손상 평균 | Corruption JEPA | Clean JEPA | 무학습 ViT |
|---|---:|---:|---:|
| 안개 | 0.422885 | 0.396777 | 0.396587 |
| 비 | 0.448021 | 0.433476 | 0.433268 |
| 모션 블러 | 0.446568 | 0.430638 | 0.427419 |
| 저조도 | 0.402309 | 0.379086 | 0.378166 |

손상 평균에서 corruption은 clean보다+0.019952(약2.00%p),random보다+0.021086(약2.11%p) 높았다.
Clean JEPA와 random은 가까운 수치였다. 작은 val의 희귀 class,한 seed,단일 head 설정으로 얻은 관측이다.
여러 seed/신뢰구간/독립 test split 검증은 하지 않았다. 실제 악천후,검출 mAP,자율주행 안전 성과는 미측정이다.

## 파일과 실행 근거

```text
/home/coder/autodrive-jepa-robustness/runs/continue10-corruption-seed17/
/home/coder/autodrive-jepa-robustness/runs/continue10-clean-seed17/
```

각 run에 `last.pt`(약377MB),`best.pt`,`encoder.pt`,`final_encoder.pt`(약87.9MB),
metrics/config/environment/data validation/summary와 selected/final evaluation JSON을 보존했다.

- `runs/paired_continuation_summary.json`: 학습 스케줄,출발/저장 checkpoint SHA-256.
- `runs/continuation_checkpoint_audit.json`: 연속2560 optimizer steps,optimizer state2560,10 validation events,선택/마지막 step.
- `runs/continuation_diagnostics_summary.json`: 네 selected/final 평가와 health.
- `data/bdd100k/labels_provenance.json`: annotation 형식/박스 수/category별 건수/데이터 검사.
- `runs/presence_probe_summary.json`: protocol,9-class별 AP,21조건,무학습 baseline,checkpoint SHA-256.
- `runs/presence-probe-seed17/`: 세 probe head,train features,모델별 결과JSON.

모든 train loss는 finite였고10 epoch validation을 완료했다. strict model loading과 source hash를 기록했다.
`final_encoder.pt`의 모든 tensor가 마지막 full checkpoint online state와 같은지 확인해 export했다.
source hash,Python AST,shell syntax,Git whitespace 검사도 통과했다.
기존5개 단위 검사는 첫 pilot 때의 결과이며 이번에 추가/재실행하지 않았다. warm-start 이후 중단/재개 bitwise 검사도 별도로 실행하지 않았다.

두 모델의 full last/selected encoder/final encoder 총6개와40개 근거 파일을 로컬에도 백업했다.
checkpoint 크기·SHA-256이 VM과 일치했다. 영수증은 로컬 `runs/continuation_local_backup.json`이다.
실제 이미지와 전체 annotation archive는 VM에 보존한다.

## 다음 실험

1. 준비된 labels와 ViT feature adapter를 YOLO neck/head에 연결하고 box mAP/recall을 측정한다.
2. 낮은 effective rank와 random에 가까운 clean probe를 점검하며 block/object mask와 representation regularization을 비교한다.
3. 전체 데이터,seed17/23/29의 동일 예산 반복,실제 악천후 holdout으로 검증 범위를 넓힌다.

현재 결과는 부분집합의 표현/객체종류 존재 성능이다. YOLO head 연결과 검출 학습은 다음 구현 단계다.
