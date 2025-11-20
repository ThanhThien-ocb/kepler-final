xuất file pdf bao gồm các nội dung này Pipeline bao gồm các giai đoạn:

Sinh instance parameters

Sinh candidate plans (RCE)

Đo latency

Build plan cover

Chuẩn bị metadata

Train SNGP Near-optimal Multihead Model

 1. Chuẩn bị môi trường
Cài môi trường Python
conda create -n sngp-env python=3.10
conda activate sngp-env
pip install -r requirements.txt

PostgreSQL + pg_hint_plan

Pipeline yêu cầu:

PostgreSQL 14+

Extension pg_hint_plan

Database đã load dữ liệu StackOverflow (hoặc dataset tương tự)

 2. Sinh Parameter Instances

Sinh parameter bindings cho query parametric (vd: q1_0).

python3 scripts/generate_params.py \
  --template stack_query_templates_with_metadata.json \
  --query-id q1_0 \
  --out data/stack_params/q1_0-*.json


Kết quả:

data/stack_params/q1_0-178214.json


File này chứa danh sách parameters dùng để sinh instance query.

🔧 3. Sinh Candidate Plans (RCE – Row Count Evolution)

Sinh các plan có cấu trúc khác nhau bằng pg_hint_plan.

python3 scripts/rce_generate_q1_0.py \
  --params-path data/stack_params/q1_0-178214.json \
  --output-dir execution_data/candidate_plans \
  --db-name stack \
  --db-user postgres \
  --db-password kelper \
  --db-host localhost \
  --max-instances 1000 \
  --max-plans-per-instance 50


Output:

execution_data/candidate_plans/
  ├── rce_q1_0_plans.json
  ├── rce_q1_0_plan_indices.json
  ├── rce_q1_0_failures.json

⏱ 4. Đo Latency cho từng Plan

Đo thời gian chạy thực tế đối với mỗi (param, plan):

python3 scripts/collect_exec_latencies.py \
  --cands execution_data/candidate_plans \
  --out artifacts/exec_latencies_q1.csv \
  --scope per-candidate \
  --repeats 3 \
  --statement-timeout-ms 45000 \
  --lock-timeout-ms 2000


Output:

artifacts/exec_latencies_q1.csv

 5. Build Plan Cover

Plan cover là tập kế hoạch nhỏ nhất bao phủ toàn bộ instance near-optimal.

python3 scripts/build_plan_cover.py \
  --lat artifacts/exec_latencies_q1.csv \
  --out artifacts/plan_cover_q1.json


Output:

artifacts/plan_cover_q1.json

📝 6. Chuẩn bị Metadata đơn giản

Metadata mô tả:

query_id

danh sách predicates

kiểu dữ liệu param

danh sách plan từ plan-cover

Ví dụ file:

{
  "query_id": "q1_0",
  "predicates": [
    { "name": "param0", "data_type": "text" },
    { "name": "param1", "data_type": "text" }
  ]
}


Lưu vào:

metadata/metadata_q1.json

🤖 7. Train SNGP Near-Optimal Multihead Model

Chạy script train:

python3 sngp_pipeline/train_sngp_nearopt.py \
  --lat artifacts/exec_latencies_q1.csv \
  --metadata metadata/metadata_q1.json \
  --plan-cover artifacts/plan_cover_q1.json \
  --out artifacts/models_q1 \
  --epochs 30 \
  --batch-size 128


Pipeline train bao gồm:

(1) Load dữ liệu

exec_latencies.csv

metadata.json

plan_cover.json

(2) Gắn DISTINCT VALUES cho các param text

Model tự rút vocabulary từ exec_latencies.csv:

param0 → danh sách site

param1 → danh sách tag

(3) Build preprocessing_config

TEXT → StringLookup → Embedding(dim=16)

INT/FLOAT → normalization / one-hot

DATE → chuyển số + chuẩn hóa

(4) Build mô hình Near-opt SNGP

MLP (64–64)

Spectral Normalization

Gaussian Process Layer (Random Fourier Features)

(5) Construct training data

X: embedding params + one-hot plan_id

Y: near-opt = 1

(6) Train

Loss: binary_crossentropy

Metric: binary_accuracy, F1, AP

(7) Lưu mô hình

Output:

artifacts/models_q1/
  ├── model.weights.h5
  ├── metadata.json  (có distinct_values)
  └── plan_cover.json

 8. Sử dụng Model để Dự đoán Plan

Ví dụ inference:

from sngp_pipeline.predict import predict_best_plan

predict_best_plan(
    model_dir="artifacts/models_q1",
    param_values=["stackoverflow", "python"]
)

 9. Cấu trúc thư mục chuẩn
project/
│
├── data/
│   └── stack_params/
│
├── execution_data/
│   └── candidate_plans/
│
├── artifacts/
│   ├── exec_latencies_q1.csv
│   ├── plan_cover_q1.json
│   └── models_q1/
│
├── metadata/
│   └── metadata_q1.json
│
├── scripts/
│   ├── generate_params.py
│   ├── rce_generate_q1_0.py
│   ├── collect_exec_latencies.py
│   └── build_plan_cover.py
│
└── sngp_pipeline/
    ├── train_sngp_nearopt.py
    ├── models.py
    ├── trainers.py
    └── predict.py
