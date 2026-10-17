# JEPA 자율주행 robustness 첫 GPU pilot 결과

실행일: 2026-10-06, Coder `owner/jwj-01`.
프로젝트: `/home/coder/autodrive-jepa-robustness`, Git branch `codex/phase1-start`.

## 완료한 작업과 범위

- 기존 연구 보고서를 보존하고 별도 프로젝트에서 mask-before-attention JEPA를 구현했다.
- 실제 BDD100K 이미지 train2,048장, val256장을 준비했다. 이미지 해시·헤더 검사 통과, split 간 scene/content 중복0개.
- corruption JEPA와 clean JEPA를 같은 초기 seed17, 모델, 데이터 순서, batch8, 각각200 optimizer steps로 학습했다.
- 모델마다1,600장의 train 이미지를 사용했다. 준비한2,048장 중 한 epoch 전체를 돌기 전 종료한 짧은 pilot이다.
- val256장 전체에 synthetic fog/rain/motion_blur/low_light × severity1–5를 평가했다. 조건마다256쌍이다.
- 두 모델의 전체 재개 상태 `last.pt`, `best.pt`와 온라인 표현 모델 `encoder.pt`를 생성했다.

무작위 초기화 ViT-S/16, 입력384×640, encoder dimension384/depth12/heads6,
encoder parameters21,958,272, predictor dimension192/depth4, visible40%, BF16을 사용했다.
RTX5090, PyTorch2.14.1+cu130 환경이다. train severity1–3, 평가severity1–5를 사용했다.

## 데이터 출처

공개 [BDD100K 이미지 미러](https://huggingface.co/datasets/karthik789338/BDD100K-Images)의
revision `8f6ba5d675567ca3a98c50947bdd83524c4749f8`을 고정했다.
train70,000개, val10,000개의 원본 경로를 인덱싱하고 각 split 안에서 별도로 무작위 추출했다.
다운로드 내용은 미러의 LFS/Git 해시와 비교했다. 공식 archive와 바이트가 같은지는 독립 검증하지 않았다.
출처·권리 notice와 선택 규칙은 [data_source.md](data_source.md)에 기록했다.

검출 annotation은 없다. 빈 boxes/labels는 `unavailable_image_only` 상태이며 객체가 없다는 정답이 아니다.
현재 실험은 이미지 표현 학습과 평가다. detection mAP, recall, 주행 안전 성능은 측정하지 않았다.

## 관측한 결과

clean-corrupt pooled embedding cosine의 severity1–5 평균이다. 같은256장과 같은 corruption 생성 규칙을 적용했다.

| 조건 | Corruption JEPA | Clean JEPA | 차이 |
|---|---:|---:|---:|
| 안개 | 0.880920 | 0.251270 | +0.629649 |
| 비 | 0.999890 | 0.999292 | +0.000599 |
| 모션 블러 | 0.999890 | 0.999963 | -0.000073 |
| 저조도 | 0.757002 | 0.623790 | +0.133212 |
| 20조건 동일 가중 평균 | 0.909425 | 0.718579 | +0.190847 |

안개 조건에서 표현 일치도 차이가 컸다. 비/모션 블러에서는 두 모델 모두 거의1로 나타나
현재 합성 손상이나 pooled metric이 차이를 구분하기 어려운지 추가 확인해야 한다.
가장 어두운 severity5에서 corruption JEPA는0.345665, clean JEPA는0.442904로 corruption 모델이 낮았다.
모든 조건에서 개선됐다는 결과는 아니다.

| 부가 지표 | Corruption JEPA | Clean JEPA |
|---|---:|---:|
| val masked prediction loss | 0.061868 | 0.119695 |
| clean pooled feature 표준편차 평균 | 0.554799 | 0.554247 |
| 서로 다른 이미지 간 cosine 평균 | 0.492026 | 0.526990 |
| 모델별 학습·검증·저장 경과 초 | 14.49 | 13.72 |

validation masked loss는 동일 val 이미지/손상 설정에서 계산했다. 두 모델의 EMA target도 각각 학습되므로
loss 차이를 검출 정확도 개선율로 해석하지 않는다. 표준편차0.001 미만이라는 단순 collapse 경고는
둘 다 발생하지 않았지만, 의미 있는 표현 학습을 입증하는 검사로 볼 수 없다.

데이터 준비454.64초, 데이터가 준비된 뒤 두 모델 학습·평가·요약159.18초.
모델별 CUDA peak allocated1,444,746,240 bytes, reserved1,595,932,672 bytes를 기록했다.
시간은 이번 작은 pilot의 실측이며 전체 데이터 학습 시간 추정으로 쓰지 않는다.

한 seed, 짧은 학습, 이미지 부분집합, 2D 합성 손상에 대한 관측이다.
실제 날씨 holdout, 여러 seed, pretrained baseline, detection fine-tuning이 남아 있다.

## 체크포인트와 재현 근거

VM run 디렉터리:

```text
/home/coder/autodrive-jepa-robustness/runs/pilot-corruption-seed17/
/home/coder/autodrive-jepa-robustness/runs/pilot-clean-seed17/
```

각 run의 `last.pt`는 약377MB이며 모델/optimizer/RNG/다음 batch/스케줄/데이터 fingerprint를 포함한다.
`encoder.pt`는 약87.9MB이며 온라인 encoder만 포함한다. `best.pt`는 heldout masked loss로 선택했다.
이번에는 validation이 마지막200-step 한 번 수행돼 best/last의 모델 상태는 동일하다.

200개 연속 optimizer step 로그, optimizer state step200, 초기값 대비 patch 가중치 변경을 확인했다.
`runs/checkpoint_audit.json`에 parameter 수, shape, step, 변경 norm이 있다.
`runs/paired_pilot_summary.json`에 모든 체크포인트 경로·크기·SHA-256과20개 조건 원시 수치가 있다.

로컬 프로젝트의 `runs/`에 두 모델의 `last.pt`, `encoder.pt`, 원시 JSON/로그/manifest를 백업했다.
27개 근거 파일과4개 체크포인트를 보존했고, 체크포인트 크기와 SHA-256이 VM 결과와 일치했다.
완료 영수증은 로컬 `runs/local_backup.json`이다.
실제 이미지는 VM의 `data/bdd100k/images/100k/`에 있다.

`last.pt`의 재개는 중단된 동일 스케줄에 한정한다. 완료된200-step 스케줄에 다른 max-steps를 지정해
연장하는 기능은 없다. 전체 학습은 전체 데이터 manifest를 명시하고 새 run으로 시작한다.

## 검사와 수정

처음 GPU 실행의 BF16 projection과 FP32 mask token dtype 불일치를 수정했다.
첫 실패 run과 trace는 VM `runs/failures/`에 남겼으며 학습 step이 진행되기 전 발생한 실패였다.
수정 후 `tests/test_semantics.py`의5개 검사 모두 통과했다.

1. masked target pixel이 visible attention과 gradient에 들어가지 않는지 검사.
2. synthetic corruption의 seed 결정성과 입력 보존 검사.
3. 이름이 달라도 같은 이미지 바이트의 train/val 중복 차단 검사.
4. 작은 CPU fixture에서 중단·재개와 연속 학습의 모든 모델 가중치 일치 검사.
5. 작은 실제 CUDA BF16 forward/backward에서 finite loss, mask token/online gradient와 target gradient 차단 검사.

이 검사는 구현의 특정 동작을 확인한다. 연구 성능의 근거는 위 실제 이미지 pilot 관측 범위로 한정한다.
원시 검사 로그: `runs/unit_checks.log`. 배포 시 source hash, Python AST, shell syntax, Git whitespace 검사도 통과했다.

## 이어서 진행할 순서

1. 전체 train/val 이미지와 detection labels를 연결한 새 manifest를 만든다.
2. 같은 예산의 충분한 JEPA/clean baseline 학습을 여러 seed로 실행한다.
3. encoder와 YOLO neck/head를 연결하고 clean 및 손상 조건의 mAP/recall을 비교한다.
4. 실제 야간·비·안개 holdout과 합성 손상 결과를 구분해 평가한다.

현재 P3/P4/P5 feature adapter는 구현돼 있으며 YOLO 검출 학습 연결은 아직 없다.
