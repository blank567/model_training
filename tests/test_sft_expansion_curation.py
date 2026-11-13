"""覆盖抽审发现的实际错误；验证保留样本文本、Mask和Token完全不变。"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import curate_sft_expansion_pool as curator
import prepare_sft_data as base


def row(prompt, response='A clear answer.', *, subset='alpaca_gpt4_zh', language='en', context='', family='alpaca_gpt4_zh', task='classification'):
    return {'id':'example', 'system':'', 'prompt':prompt, 'response':response, 'input_context':context,
            'language':language, 'task':task, 'source':{'source_subset':subset, 'reserve_family':family},
            'input_ids':[4,5,2], 'loss_mask':[0,1,1]}


@pytest.mark.parametrize('prompt,expected', [
    ('How can I evaluate a classification model?', curator.OTHER),
    ('Tell me some information about farming.', curator.OTHER),
    ('A secret agent was extracted quietly. What happens next?', curator.OTHER),
    ('Explain the emotion in this poem.', curator.OTHER),
    ('Classify the following words: cat, run.', 'classification'),
    ('请分类以下单词：cat、run。', 'classification'),
    ('请找出以下句子中的夸张表达：大学应该禁止他。', 'extraction'),
    ('请用JSON输出结果。', 'format_following'),
])
def test_action_taxonomy_does_not_match_arbitrary_substrings(prompt, expected):
    assert curator.strict_task(row(prompt)) == expected


def test_missing_material_clarification_is_not_grounded_or_summary():
    item = row('Summarize my lecture.', 'Please provide the lecture text first.', subset='smol-summarize-20k')
    assert curator.strict_task(item) == curator.OTHER


def test_keyword_counts_whole_words_instead_of_importance_substring():
    prompt = "the word 'important' should appear at least 3 times."
    assert curator.explicit_constraints(prompt, 'important importance important', 'en') == 'required_keyword_frequency_violated'
    assert curator.explicit_constraints(prompt, 'important important important', 'en') is None


def test_combined_explicit_requirements_and_chinese_per_item_budget():
    prompt = 'At least 2 placeholders. Include keywords [climate, environment]. The word [sustainable] should appear at least 2 times. Add a postscript starting with P.S.'
    answer = '[climate] [environment] sustainable sustainable\nP.S. Start today.'
    assert curator.explicit_constraints(prompt, answer, 'en') is None
    assert curator.explicit_constraints(prompt, answer.replace('[climate]', 'climate'), 'en') == 'explicit_placeholder_budget_violated'
    assert curator.explicit_constraints('每个步骤的说明严格限制在15-20字之间。', '1. 说明:太短。', 'zh') == 'per_item_chinese_character_budget_violated'


def test_oasst_requires_individual_accept_and_brainstorming_is_not_blanket_blocked():
    item = row('How can I plan my week?', family='oasst2')
    assert curator.quarantine_reason(item, {}) == 'oasst_not_individually_reviewed_for_this_release'
    assert curator.quarantine_reason(item, {'example':{'decision':'accept'}}) is None
    clean = row('请提出家庭节水产品的创新想法。', '可以考虑循环用水装置。', subset='brainstorming.parquet', language='zh')
    assert curator.quarantine_reason(clean, {}) is None
    clean['prompt'] += '\n\n当然可以，以下是完整答案。'
    assert curator.quarantine_reason(clean, {}) == 'user_prompt_contains_assistant_role_or_answer'


def test_unverified_url_is_quarantined_without_fetching():
    item = row('Summarize https://example.com/article')
    assert curator.quarantine_reason(item, {}) == 'external_material_not_fetched_or_verified'


def test_curated_pool_preserves_original_evidence_and_token_payload(tmp_path):
    data = tmp_path/'data'; data.mkdir(); work = tmp_path/'work'; work.mkdir()
    keep = row('Explain a classification model.', task='classification')
    reject = {**row('A second question.'), 'id':'excluded'}
    base.write_jsonl(data/'candidates.jsonl', [keep,reject])
    parent = base.sha256(data/'candidates.jsonl')
    base.atomic_json(data/'pool_manifest.json', {'candidate_count':2, 'candidates_sha256':parent})
    base.write_jsonl(data/'review_sample.jsonl', [{'id':'example'},{'id':'excluded'}])
    path = work/'decisions.json'
    base.atomic_json(path, {'pool_sha256':parent, 'test_content_used':False, 'reviewed':[
        {'id':'example','decision':'accept','reason':'Clear explanation.'},
        {'id':'excluded','decision':'exclude','reason':'Unsupported answer.'}]})
    result = curator.curate(data, path, work, min_free=0)
    published = list(curator.row_stream(data/'candidates.jsonl'))
    assert result['candidate_count'] == 1 and published[0]['task'] == curator.OTHER
    assert {key:value for key,value in published[0].items() if key!='task'} == {key:value for key,value in keep.items() if key!='task'}
    assert base.sha256(data/'candidates.stage1.jsonl') == parent
    review = json.loads((work/'review_decisions.json').read_text(encoding='utf-8'))
    assert review['parent_pool_sha256']==parent and review['pool_sha256']==base.sha256(data/'candidates.jsonl')
    with pytest.raises(ValueError, match='re-curate'):
        curator.curate(data, path, work, min_free=0)
