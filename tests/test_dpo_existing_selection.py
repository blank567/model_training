from pathlib import Path
from collections import Counter
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finalize_existing_reading_dpo import select_existing


def rows(prefix,zh,en):return [{'id':prefix+str(i),'language':'zh' if i<zh else 'en'} for i in range(zh+en)]


def test_existing_pairs_keep_equal_count_language_counts_and_no_duplicates():
    old=rows('old',3,2);new=rows('new',5,5);left,right=select_existing(old,new)
    assert left==old and len(right)==5
    assert Counter(r['language'] for r in right)==Counter(r['language'] for r in old)
    assert len({r['id'] for r in right})==5 and select_existing(old,new)[1]==right


def test_missing_language_candidates_are_not_fabricated():
    with pytest.raises(ValueError,match='Insufficient'):select_existing(rows('old',3,2),rows('new',2,5))
