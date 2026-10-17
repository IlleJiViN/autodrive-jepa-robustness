# 이전 작업을 새 프로젝트로 연결한 기록

새 시작: 2026-10-06, Coder `owner/jwj-01`, `/home/coder/autodrive-jepa-robustness`.

이전 연구 제목: Natural Corruption-aware JEPA 기반 자율주행 강건성.
원본 보고서 chat: `01a10060-94c4-7eb0-a6bd-b9563c66c58c`.
후속 scaffold chat: `01a10b05-1c9f-7b00-aa12-46061babe4ec`.

이전 자율주행 폴더에는 한국어 설계 보고서와 실행 전 Phase 1 코드 골격이 있었다.
주행 데이터와 학습 체크포인트/성과 수치는 없었다. 이전 학습 시간 표는 실측이 아닌 추정이었다.
원본 DOCX는 reference/research_report.docx에 보존한다. 원본 보고서는 페이지 렌더링 검수가 확인되지 않았다.

2026-10-06 VM 조회:
- owner/jwj-01: RTX5090, CCTV/CHAD temporal pose+scene checkpoint64개와 결과JSON2개가 확인됐다.
- owner/ai-lab: graph GATv2 JEPA checkpoint가 확인됐다.
- 이 파일들을 새 자율주행 ViT-JEPA의 학습 성과로 취급하지 않는다.

이전 코드의 API와 파일을 그대로 실행하는 대신 새 루트에 mask-before-attention, split검사,
저장/재개, EMA target, 실제 데이터 변환기, 표현 평가를 다시 작성했다.
2026-10-06 사용자 `ㄱㄱ` 지시로 데이터 다운로드, 실제 GPU 학습, 동작 검사를 진행했다.
BDD100K pilot 이미지2,304장 준비와 두 모델의200-step 학습·평가를 완료했다.
이번에 새로 생성된 자율주행 체크포인트/성과의 근거는 `docs/pilot_results.md`와 `runs/paired_pilot_summary.json`이다.
