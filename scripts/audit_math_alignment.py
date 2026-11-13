"""独立验收已发布的数学题库、DPO视图与测试集。"""
import argparse
import json
from llm_math_data.audit import audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), ensure_ascii=False, indent=2))
