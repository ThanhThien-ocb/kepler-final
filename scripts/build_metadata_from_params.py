#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_metadata_from_params.py
-----------------------------

Sinh file metadata (predicates + distinct_values cho các cột text)
từ file params kiểu SPQD/Kepler, ví dụ: q1_0-178214.json

Cấu trúc input (ví dụ):

{
  "q1_0": {
    "query": "...",
    "predicates": [
      {
        "alias": "site",
        "column": "site_name",
        "operator": "=",
        "data_type": "text",
        "table": "site"
      },
      {
        "alias": "tag",
        "column": "name",
        "operator": "=",
        "data_type": "text",
        "table": "tag"
      },
      ...
    ],
    "params": [
      { "@param0": "stackoverflow.com", "@param1": "java" },
      { "@param0": "superuser.com",     "@param1": "php"  },
      ...
    ]
  }
}

Output metadata tối thiểu:

{
  "template_id": "q1_0",
  "num_instances": 178214,
  "predicates": [
    {
      "index": 0,
      "alias": "site",
      "column": "site_name",
      "operator": "=",
      "data_type": "text",
      "table": "site",
      "distinct_values": ["stackoverflow.com", "superuser.com", ...]
    },
    ...
  ]
}

File này sau đó sẽ được nạp vào TrainerBase (metadata["predicates"]).
"""

import argparse
import json
from typing import Any, Dict, List, Tuple


def load_template_obj(path: str) -> Tuple[str, Dict[str, Any]]:
    """Đọc file params JSON và trả về (template_id, template_obj)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and len(data) == 1:
        # dạng {"q1_0": {...}}
        template_id = next(iter(data.keys()))
        template_obj = data[template_id]
    else:
        raise ValueError(
            f"File {path} không ở dạng {{'qX_Y': {{...}}}}. "
            f"Vui lòng kiểm tra lại cấu trúc."
        )

    if "predicates" not in template_obj or "params" not in template_obj:
        raise ValueError(
            f"template_obj không có 'predicates' hoặc 'params'. "
            f"Các key hiện có: {list(template_obj.keys())}"
        )

    return template_id, template_obj


def infer_distinct_values(
    template_obj: Dict[str, Any],
    max_values: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Thêm trường 'distinct_values' cho từng predicate kiểu text,
    dựa trên danh sách 'params' trong template_obj.
    """
    predicates: List[Dict[str, Any]] = template_obj["predicates"]
    params_list: List[Dict[str, Any]] = template_obj["params"]

    param_count = len(predicates)
    print(f"[INFO] Số predicate = {param_count}")
    print(f"[INFO] Số params instance = {len(params_list)}")

    # Chuẩn bị set để gom distinct cho từng param index
    distinct_sets: List[set] = [set() for _ in range(param_count)]

    for i, param_row in enumerate(params_list):
        # Mỗi param_row dạng {"@param0": "...", "@param1": "...", ...}
        for idx in range(param_count):
            key1 = f"@param{idx}"
            key2 = f"param{idx}"  # fallback nếu dùng tên param0 thay vì @param0

            if key1 in param_row:
                val = param_row[key1]
            elif key2 in param_row:
                val = param_row[key2]
            else:
                continue

            # Lưu giá trị string hoá (để ổn định)
            if val is None:
                s = ""
            else:
                s = str(val)

            distinct_sets[idx].add(s)

        if (i + 1) % 20000 == 0:
            print(f"[INFO] Đã xử lý {i + 1} dòng params...")

    # Gắn distinct_values vào predicates
    enriched_preds: List[Dict[str, Any]] = []
    for idx, pred in enumerate(predicates):
        pred_copy = dict(pred)  # tránh mutate bản gốc
        dtype = pred_copy.get("data_type")

        if dtype == "text":
            values = sorted(distinct_sets[idx])
            if max_values is not None and len(values) > max_values:
                print(
                    f"[WARN] Predicate index {idx} có {len(values)} distinct text values, "
                    f"cắt xuống còn {max_values}."
                )
                values = values[:max_values]
            pred_copy["distinct_values"] = values
            print(
                f"[INFO] Predicate {idx} (text) có {len(values)} distinct_values "
                f"(sau khi cắt nếu có)."
            )
        else:
            # non-text: không cần distinct_values
            print(
                f"[INFO] Predicate {idx} (data_type={dtype}) không cần distinct_values."
            )

        # đảm bảo có index
        pred_copy.setdefault("index", idx)
        enriched_preds.append(pred_copy)

    return enriched_preds


def main():
    parser = argparse.ArgumentParser(
        description="Sinh metadata (predicates + distinct_values) từ file params."
    )
    parser.add_argument(
        "--params-path",
        required=True,
        help="Đường dẫn tới file params JSON (vd: data/stack_params/q1_0-178214.json)",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Đường dẫn file output metadata JSON (vd: artifacts/metadata_q1_0.json)",
    )
    parser.add_argument(
        "--max-distinct-text",
        type=int,
        default=None,
        help=(
            "Giới hạn số lượng distinct_values cho mỗi predicate text. "
            "Nếu None, giữ nguyên tất cả."
        ),
    )

    args = parser.parse_args()

    template_id, template_obj = load_template_obj(args.params_path)
    enriched_preds = infer_distinct_values(
        template_obj, max_values=args.max_distinct_text
    )

    metadata: Dict[str, Any] = {
        "template_id": template_id,
        "num_instances": len(template_obj.get("params", [])),
        "predicates": enriched_preds,
    }

    # Bạn có thể thêm các trường khác nếu cần, ví dụ:
    # metadata["query"] = template_obj.get("query")

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Đã ghi metadata vào: {args.out_path}")
    print(f"[DONE] template_id={metadata['template_id']}, "
          f"num_instances={metadata['num_instances']}, "
          f"num_predicates={len(metadata['predicates'])}")


if __name__ == "__main__":
    main()
