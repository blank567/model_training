"""从固定公开 MathInstruct 快照准备数学 SFT 数据，不启动 GPU。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_math_public.build import main

if __name__ == '__main__':
    main()
