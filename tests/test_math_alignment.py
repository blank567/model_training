"""数学数据的有效性、奖励抗绕过和测试隔离；全部在CPU执行。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from llm_math_data.audit import audit, rows
from llm_math_data.build import build, manifest_identity, read_json, sha256, write_json
from llm_math_data.problems import TEXT
from llm_math_data.verifier import accuracy_reward, expression_key, expression_value, numeric_value, reward


@pytest.mark.parametrize('answer', ['18 and 19', '18\n19', 'The answer is 18', '```18```',
                                  '<answer>18</answer>', '2*9', 'NaN', '1/0', '1e99999'])
def test_reward_does_not_accept_format_hacks_or_multiple_answers(answer):
    assert reward(answer, '18') == 0


def test_exact_numbers_and_conversational_reward_adapter():
    assert reward(' 18\n', '18') == 1
    assert reward('1,000.0', '1000') == 1
    assert reward('0.5', '1/2') == 1
    assert reward('10,00', '1000') == 0
    assert reward('b', 'B', 'choice_exact') == 1
    assert accuracy_reward([[{'role': 'assistant', 'content': '18'}], '19'], ['18', '18']) == [1., 0.]
    assert numeric_value('1/0') is None


@pytest.mark.parametrize('expression', ['__import__("os")', '2**100000', 'a+1', '(1).__class__'])
def test_independent_math_solver_rejects_non_arithmetic_code(expression):
    with pytest.raises(ValueError):
        expression_value(expression)


def test_commutative_expression_identity_does_not_create_fake_new_problems():
    assert expression_key('(2+3)*4') == expression_key('(3+2)*4')
    assert expression_key('5-2') != expression_key('2-5')
    assert expression_value('(12+8)*3') == 60


@pytest.fixture
def math_release(tmp_path):
    from llm_lab.tokenization.bpe import train_bpe
    root = tmp_path
    cache = root / 'cache'
    cache.mkdir()
    tokenizer_path = root / 'tokenizer.json'
    texts = [text for family in TEXT.values() for variants in family.values() for text in variants]
    tokenizer = train_bpe(texts + ['1234567890'], tokenizer_path, vocab_size=300, min_frequency=1)
    context = {'question_hashes': [], 'tokenizer': {'vocab_size': tokenizer.get_vocab_size(),
               'sha256': sha256(tokenizer_path), 'pad_id': 0, 'unk_id': 1, 'eos_id': 2},
               'tokenization': {'empty_system_prefix_template': 'User:\n{prompt}\nAssistant:\n',
                                'prefix_template': 'System:\n{system}\nUser:\n{prompt}\nAssistant:\n'}}
    write_json(cache / 'project_exclusions.json', context)
    source = cache / 'gsm_test.jsonl'
    source.write_text(json.dumps({'question': 'What is 2 plus 5?', 'answer': '#### 7'}) + '\n' +
                      json.dumps({'question': 'long question ' * 300, 'answer': '#### 1'}) + '\n', encoding='utf-8')
    (cache / 'openai__grade-school-math').mkdir()
    (cache / 'openai__grade-school-math/LICENSE').write_text('MIT test fixture', encoding='utf-8')
    (cache / 'tal_hf_dataset_card.md').write_text('license: mit; test fixture', encoding='utf-8')
    write_json(cache / 'sources.lock.json', {'fixture': True, 'sources': [
        {'repository': 'openai/grade-school-math', 'revision': '0' * 40, 'path': 'gsm_test.jsonl',
         'sha256': sha256(source), 'upstream_path': 'data/test.jsonl', 'role': 'external_test_only'}]})
    config = {'project_root': '.', 'seed': 42, 'fixture': True,
              'source_cache': 'cache', 'tokenizer': 'tokenizer.json',
              'output_dir': 'artifacts/posttrain/math_fixture',
              'counts': {'train': 32, 'validation': 16, 'test_id': 16, 'test_template': 16},
              'nested_training_examples': [16]}
    path = root / 'build.json'
    write_json(path, config)
    result = build(path)
    return Path(result['manifest']), result, path


def test_complete_release_checks_bilingual_views_nested_data_and_external_denominator(math_release):
    manifest_path, result, config = math_release
    verified = audit(manifest_path)
    assert verified['status'] == 'passed'
    assert verified['splits']['train']['examples'] == 32
    assert verified['splits']['train']['groups'] == 16
    assert verified['splits']['train']['zh_answer_token_fraction'] == .5
    assert result['external_tests']['gsm8k_en'] == {'eligible': 1, 'official': 2}
    assert result['external_exclusions'] == 1
    manifest = read_json(manifest_path)
    assert manifest['fixture'] is True and manifest['training_started'] is False
    with pytest.raises(FileExistsError):
        build(config)


def rewrite_record(manifest_path, file_key, mutation):
    manifest = read_json(manifest_path)
    path = manifest_path.parent / manifest['files'][file_key]['path']
    data = rows(path)
    mutation(data)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in data), encoding='utf-8')
    manifest['files'][file_key].update(sha256=sha256(path), bytes=path.stat().st_size)
    manifest['data_identity'] = manifest_identity(manifest)
    write_json(manifest_path, manifest)


def test_independent_audit_catches_wrong_preference_even_if_file_hash_is_updated(math_release):
    manifest_path, _, _ = math_release
    rewrite_record(manifest_path, 'dpo/train', lambda data: data[0].update(rejected=data[0]['chosen']))
    with pytest.raises(ValueError, match='chosen/rejected'):
        audit(manifest_path, require_acceptance=False)


def test_heldout_template_cannot_be_replaced_by_a_training_template(math_release):
    manifest_path, _, _ = math_release
    rewrite_record(manifest_path, 'rlvr/test_template', lambda data: data[0]['metadata'].update(template_id=0))
    with pytest.raises(ValueError, match='Template isolation'):
        audit(manifest_path, require_acceptance=False)


def test_checksum_failure_is_not_hidden_by_prior_acceptance(math_release):
    manifest_path, _, _ = math_release
    manifest = read_json(manifest_path)
    path = manifest_path.parent / manifest['files']['rlvr/validation']['path']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        audit(manifest_path)
