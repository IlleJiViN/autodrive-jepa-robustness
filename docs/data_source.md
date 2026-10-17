# 2026-10-06 실제 pilot 데이터

원본 데이터: BDD100K / Berkeley DeepDrive (Yu et al., CVPR 2020).
공식 다운로드 안내: https://github.com/SysCV/bdd100k-models/blob/main/doc/PREPARE_DATASET.md

VM에서 기존 ETH 다운로드 도메인은 DNS 해석 실패, Berkeley 도메인은 TLS hostname 오류가 발생했다.
공개 원본 이미지 미러를 사용한다: https://huggingface.co/datasets/karthik789338/BDD100K-Images
고정 revision: `8f6ba5d675567ca3a98c50947bdd83524c4749f8`.

미러는 research/educational/nonprofit 이용을 명시하고 원래 BDD100K copyright/license notice를 제공한다.
notice와 README는 `data/bdd100k/SOURCE_LICENSE_NOTICE.md`, `SOURCE_README.md`로 보존한다.
공식 archive와 이미지 바이트가 완전히 같은지는 검증하지 않았다.

원본 경로 `images/100k/train` 70,000개와 `images/100k/val` 10,000개를 전부 인덱싱했다.
train2,048장(seed17), val256장(seed18)을 각각 원래 split 안에서 무작위 추출한다.
원본 frame 이름을 scene_id로 보존하고 미러의 LFS SHA-256/Git blob hash와 다운로드 내용을 비교한다.
매 image SHA-256을 manifest에 넣으며 학습 시작 시 검사한다. train/val scene_id 및 이미지 바이트 중복을 차단한다.

**현재 `data/manifests/bdd100k_train.jsonl`, `bdd100k_val.jsonl`은 위 pilot 부분집합이다. 전체 80,000장 manifest가 아니다.**
전체 데이터 실험 전에 새 원본 manifest를 준비하고 config의 경로를 명시적으로 변경해야 한다.

이미지 미러에는 detection annotation이 없다. 첫 image-only manifest의 boxes/labels는 `annotation_status=unavailable_image_only`를 동반한 빈 값이다.
이 값은 객체가 없다는 ground truth가 아니다. JEPA loss는 image-only이므로 해당 pilot을 실행했다.

데이터 준비 결과/수량/해시 근거: `data/bdd100k/pilot_provenance.json`.
실제 모델/평가 결과: `runs/paired_pilot_summary.json`. 하나의 seed로 200 steps씩 실행한 pilot 결과로 한정한다.

## 추가로 확보한 legacy detection labels

공개 BDD download endpoint의 `http://128.32.162.150/bdd100k/bdd100k_labels.zip`을 받았다.
189,638,612 bytes, SHA-256 `7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a`.
archive에는 train70,000/val10,000/test20,000개 image별 JSON이 있다.
legacy `frames[0].objects` 형식이며 Detection2020 trainval annotation으로 표기하지 않는다.
HTTP로 받은 보존 파일의 SHA-256과 selected JSON의 ZIP CRC를 기록했다. source published hash는 없었다.
권리 notice는 `data/bdd100k/annotations/SOURCE_LICENSE.txt`로 보존했다.

basename과 원본 split으로 기존 pilot train2,048장/val256장 전체와 연결했다.
person→pedestrian, bike→bicycle, motor→motorcycle alias를 canonical10 classes에 매핑했다.
비검출 object와 잘못된 면적의 box train15개/val3개를 제외하고 건수를 기록했다.
최종 박스 train37,868개, val4,978개. 이미지 SHA-256/box 범위/scene 및 content split 검사를 통과했다.

새 manifest는 `bdd100k_train_labeled.jsonl`, `bdd100k_val_labeled.jsonl`이다.
기존 image-only manifest는 유지해 완료된 JEPA run의 fingerprint를 보존했다.
YOLO 형식 image symlink/normalized labels와 `data/bdd100k/yolo_pilot/dataset.yaml`도 생성했다.
YOLO 검출 학습/box mAP 측정은 아직 하지 않았다.

근거: `data/bdd100k/labels_provenance.json`, `runs/presence_probe_summary.json`, [continuation_results.md](continuation_results.md).
