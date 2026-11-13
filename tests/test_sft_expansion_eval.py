"""辅助评测必须来自训练与抽审都未使用的候选。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_sft_expansion_eval import unused_candidates


def test_auxiliary_candidates_exclude_both_training_and_reviewed_examples():
    rows=[{'id':value} for value in ('train','review-accepted','review-rejected','unused')]
    assert unused_candidates(rows,{'train'},{'review-accepted','review-rejected'})==[{'id':'unused'}]


def test_duplicate_candidate_ids_are_not_silently_reused():
    with pytest.raises(ValueError,match='Repeated'):
        unused_candidates([{'id':'same'},{'id':'same'}],set(),set())
