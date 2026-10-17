# 객체 단위 JEPA를 검증하는 수정 파일럿

2026-10-07. 이번 구현의 작업 전제는 **전체 장면을 보면서 YOLO 박스로 정한 객체 영역을 JEPA의 예측 단위로 사용한다**는 것이다. 기본 ViT 격자는 patch16으로 유지한다. 박스가 포함하는 격자 패치들을 한 객체 영역으로 묶고, 그 영역 전체를 가린 다음 객체의 clean latent를 예측한다. 입력을 객체 crop으로 잘라 주변 장면을 제거하는 구조는 아니다.

기존 파일럿은 uniform random mask와 custom frozen detector probe를 사용했다. 따라서 기존 mAP 순위는 이 객체 단위 가설을 검증한 결과가 아니다. `YOLOFeatureAdapter`라는 이름도 실제 YOLO가 생성한 박스를 사전학습에 넣었다는 뜻이 아니다.

## 구현된 경로

1. 지정한 기존 YOLO checkpoint로 clean 이미지의 박스를 생성하고 checkpoint SHA-256, confidence, 좌표계, package version을 남긴다. 박스는 객체 영역을 정하는 지침이다. teacher의 class 예측은 manifest에 출처 정보로 저장되지만 JEPA/MAE 사전학습 loss에는 들어가지 않는다.
2. 같은 이미지의 손상 view를 context encoder에 넣는다. 표본·epoch에 의해 결정되는 mask로 객체 영역 전체를 attention 전에 제거한다. 각 이미지에서 최대 네 객체를 선택하며, 빈 박스 또는 너무 큰 박스에는 명시적 random fallback을 적용한다.
3. 가려진 token 수는 전체 960개 중 576개로 고정하고, 객체를 가린 뒤 남는 예산은 무작위 패치로 채운다. 전체 화면 좌표와 보이는 주변 문맥은 유지한다.
4. JEPA는 clean EMA target의 패치 latent와 객체 영역을 면적 가중 평균한 latent를 예측한다. 픽셀 baseline은 같은 mask와 같은 객체 영역 지침 아래 clean normalized pixel을 예측하고 객체별 pixel error를 추가한다.

두 구현은 공식 I-JEPA/MAE 재현이 아닌 연구용 custom objective다. YOLO 지침은 외부 사전학습 검출기의 정보를 사용하는 추가 지침이다. 사람 박스로 대체한 초기 config는 **oracle-box control**이며 실제 YOLO proposal 결과와 구분한다. production 경로의 효과를 주장하려면 실제 proposal 경로를 실행해야 한다.

## 비교군

| 조건 | mask | 추가 객체 목적 | 확인하는 효과 |
|---|---|---|---|
| random JEPA | 무작위 | 없음 | 기존 종류의 latent 기본선 |
| object JEPA | 동일 YOLO 박스 지침 | pooled 객체 latent 정렬 | 제안 가설 |
| random masked autoencoder | 무작위 | 없음 | 픽셀 기본선 |
| object masked autoencoder | 동일 YOLO 박스 지침 | 객체별 pixel error | 박스 정보를 공유한 목적함수 비교 |
| mask-only JEPA, 선택 | 동일 YOLO 박스 지침 | 없음 | mask와 객체 loss의 효과 분리 |

같은 ViT-S, 2,048 train / 256 validation, image size, seed, corruption, batch, optimizer schedule, 2,560 step을 사용한다. 사전학습은 새 초기화에서 시작하고 final encoder를 사용한다. seed17은 작동 및 탐색 파일럿이고 seed23/29 반복과 별도 holdout이 필요하다. JEPA와 pixel baseline의 loss 숫자를 서로 성능으로 비교하지 않는다. teacher proposal 생성 비용과 사전학습 비용은 실제 측정 후 포함한다.

## 의미 보존 평가

`run_object_probe.py`는 **알려진 GT 박스가 주어진 객체 분류**를 평가한다. 원본 이미지 전체를 encoder에 넣고 각 객체의 면적 가중 feature를 읽는다. clean train feature로 동일한 선형 분류기를 학습한 뒤 clean 및 네 손상 × severity1–5를 비교한다.

- 클래스별 recall, macro AP, balanced accuracy, ECE, Brier score와 평가 객체 수.
- 야간, resized box area가 32² pixel 미만인 작은 객체, occluded 객체의 별도 결과.
- clean에서 맞힌 객체가 손상 후에도 맞는 비율과 clean-corrupt 객체 feature cosine.
- 객체 embedding의 effective rank, 미학습 encoder, 박스 위치·크기만 사용하는 geometry baseline. cosine이 높다는 이유만으로 의미 보존이라고 판정하지 않는다.

평가의 GT 위치는 표현의 정보량을 분리해 보는 oracle 조건이다. 이 점수는 객체가 검출되지 않는 오류를 포함하지 않으며 box mAP, 추적 성능, 실제 악천후 전이 또는 안전한 주행 행동을 의미하지 않는다. pseudo class를 semantic 정답으로 사용하는 것은 코드에서 거부한다.

severity4/5에서는 관측 단서가 사라질 수 있다. 이 파일럿은 원 GT box/class를 유지한 stress 평가이며 손상 전후의 완전한 관측 가능성을 보장하지 않는다. 행동 단계에서는 visibility와 불확실성을 별도로 다뤄야 한다.

## 행동 파일럿 연결

원 연구의 후단은 객체·추적·ego 상태를 받는 driving-specific contrastive decision model이다. 현재 BDD 정지 이미지/박스 manifest에는 실제 행동 정답, ego 상태, 객체 시간 이력이나 counterfactual 결과가 없다. 박스 위치로 임의의 brake/yield 정답을 만들어 실제 행동 성과로 보고하지 않는다.

후속 실행에는 CARLA의 동일 route/scenario seed에서 expert action과 ego/track 상태를 수집하거나, 실제 행동 라벨이 있는 주행 로그가 필요하다. 같은 정책 구조와 학습 예산을 각 encoder 뒤에 연결하고 collision/near-miss, yield, traffic violation, route completion을 함께 평가한다. 입력 손상만 바꾸는 paired 실행과 보이지 않는 객체의 visibility/uncertainty 처리를 포함한다. 이 단계는 현재 실행되지 않았다.

## 실행 명령

VM에서 프로젝트 root로 이동한 뒤 지정한 기존 detector weights를 사용한다.

```bash
/home/coder/venv/bin/python tools/export_yolo_object_targets.py \
  --weights /absolute/path/to/existing/yolo.pt \
  --output-dir data/manifests/yolo-object-targets

/home/coder/venv/bin/python tools/run_object_pilot.py \
  --training-config data/manifests/yolo-object-targets/training_config.json \
  --semantic-config configs/object_pilot.json \
  --run runs/object-yolo-seed17 --seed 17 --steps 2560 \
  --include-mask-only-jepa
```

YOLO 오류를 제거한 oracle control은 `--training-config configs/object_pilot.json`으로 별도 새 run에 실행한다. 중단된 개별 학습은 동일 config/objective/mask/seed/step schedule로 `run_object_pretraining.py --resume .../last.pt`를 사용한다. 기존 run 또는 결과를 덮어쓰지 않는다.

## 현재 검증과 실행 상태

- 로컬 RTX 4060 Ti, PyTorch2.14.0+cu130에서 객체 경계, mask 예산, RNG 독립성, hidden pixel 누출 차단, 두 objective의 CUDA BF16 gradient, 중단/재개 가중치 일치, collapsed classifier 실패 판정과 YOLO/GT 출처 분리를 검사했다. 8개 기능 검사가 통과했다. 작은 합성 fixture 및 변환 mock의 검사이며 연구 성과나 실제 YOLO inference 검증이 아니다.
- 실제 주석 37,868 train box / 4,978 validation box의 mask geometry audit를 실행했다. epoch0/seed17에서 train2,048장과 validation256장 모두 최소 한 객체 영역을 선택했고, 평균 약 네 객체를 선택했다. 항상 visible384/masked576 token으로 맞췄다. 이미지와 모델 예측을 사용한 성능 평가가 아니다.
- 2026-10-07 Coder 접속은 endpoint DNS 오류로 실패했다. 로컬 Tailscale CLI는 `NoState`를 반환하며 서비스 재시작도 접근 권한 오류로 실패했다. raw 이미지가 VM에 있어 새 proposal 생성, 연구 사전학습, 실제 데이터 semantic probe를 실행하지 못했다.
- 새 연구 checkpoint 및 실제 의미/행동 성과: **없음**. 이전 checkpoint/성과는 기존 파일에 보존되어 있다.

배경 자료: [SemMAE](https://papers.nips.cc/paper/2022/hash/5c186016d0844767209dc36e9e61441b-Abstract-Conference.html)는 semantic-guided mask가 MAE에도 적용될 수 있음을 보여준다. 따라서 객체 지침의 효과를 JEPA만의 우위로 해석하지 않고 동일 지침을 두 objective에 공유한다. YOLO proposal API는 [공식 Predict 문서](https://docs.ultralytics.com/modes/predict/)를 따르며 실제 detector inference는 연결 복구 후 확인해야 한다.
