"""准备独立中英数学偏好与可验证奖励数据；不会启动模型训练。"""
import argparse
import json
from llm_math_data.build import build


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = build(args.config)
    print(json.dumps({key: value for key, value in result.items() if key != 'acceptance'},
                     ensure_ascii=False, indent=2))
