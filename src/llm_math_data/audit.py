"""从发布文件独立重算答案、Token、分区和来源，拒绝仅信任生成器计数。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .build import code_identity, manifest_identity, prefix, read_json, sha256, token_stats
from .problems import FAMILIES
from .verifier import expression_key, expression_value, numeric_value, question_hash, reward


def safe_path(folder, value):
    path = folder / value
    if Path(value).is_absolute() or '..' in Path(value).parts or path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError('Release path escaped its directory')
    return path


def rows(path):
    import json
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def audit(manifest_path, require_acceptance=True):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    manifest_path = Path(manifest_path).resolve()
    folder = manifest_path.parent
    manifest = read_json(manifest_path)
    if manifest.get('kind') != 'bilingual_arithmetic_alignment' or manifest_identity(manifest) != manifest.get('data_identity'):
        raise ValueError('Math manifest identity mismatch')
    if manifest['data_code_sha256'] != code_identity():
        raise ValueError('Math data builder/verifier/audit source changed')
    if manifest['training_policy'] != {'epochs': 1, 'stop_epochs': None}:
        raise ValueError('Math training policy is not one epoch')
    for name, digest in manifest['provenance_files'].items():
        if sha256(safe_path(folder, 'provenance/' + name)) != digest:
            raise ValueError('Data provenance changed')
    converter = OpenCC('t2s')
    tokenizer_spec = manifest['tokenizer']
    tokenizer_path = safe_path(folder, tokenizer_spec['path'])
    if sha256(tokenizer_path) != tokenizer_spec['sha256']:
        raise ValueError('Math tokenizer identity mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id('<|eos|>') != 2 or tokenizer.get_vocab_size() != tokenizer_spec['vocab_size']:
        raise ValueError('Unexpected tokenizer/EOS contract')
    exclusions = set(read_json(folder / 'provenance/project_exclusions.json')['question_hashes'])
    for name, spec in manifest['files'].items():
        path = safe_path(folder, spec['path'])
        if sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('Data file checksum mismatch: ' + name)
    all_ids, all_groups, all_expressions, all_questions = set(), set(), set(), set()
    split_reports = {}
    for split, expected in manifest['synthetic_splits'].items():
        data = rows(folder / manifest['files']['rlvr/' + split]['path'])
        preferences = rows(folder / manifest['files']['dpo/' + split]['path'])
        if len(data) != expected['examples'] or len(preferences) != len(data):
            raise ValueError('Actual record count differs from manifest')
        reference = {row['id']: row for row in data}
        if len(reference) != len(data) or {row['id'] for row in preferences} != set(reference):
            raise ValueError('DPO and RLVR row IDs are not a one-to-one match')
        ids, groups, expressions, questions = set(), set(), set(), set()
        group_rows = {}
        maximum = 0
        language_answers = Counter()
        for row in data:
            meta = row['metadata']
            if row['language'] not in {'zh', 'en'} or meta['family'] not in FAMILIES or meta['split'] != split:
                raise ValueError('Problem language/family/split changed')
            allowed = {0, 1} if split in {'train', 'test_id'} else {2} if split == 'validation' else {3, 4}
            if meta['template_id'] not in allowed:
                raise ValueError('Template isolation violated')
            answer = expression_value(meta['expression'])
            if answer.denominator != 1 or numeric_value(row['ground_truth']) != answer:
                raise ValueError('Independent arithmetic verification failed')
            if expression_key(meta['expression']) != meta['expression_key']:
                raise ValueError('Expression group fingerprint changed')
            if row['group_id'] != 'arith_' + meta['expression_key'][:24]:
                raise ValueError('Group identity changed')
            if converter.convert(row['question']) != row['question'] or converter.convert(row['prompt']) != row['prompt']:
                raise ValueError('Chinese contains non-normalized traditional characters')
            digest = question_hash(row['question'], converter)
            if digest in exclusions or question_hash(row['prompt'], converter) in exclusions:
                raise ValueError('Known project question overlap')
            ids.add(row['id']); groups.add(row['group_id']); expressions.add(meta['expression_key']); questions.add(digest)
            group_rows.setdefault(row['group_id'], []).append(row)
        for pair in preferences:
            original = reference[pair['id']]
            for key in ('group_id', 'language', 'question', 'prompt', 'metadata', 'source', 'token_counts'):
                if pair[key] != original[key]:
                    raise ValueError('DPO/RLVR question view changed')
            if (pair['pair_origin'] != 'programmatic_counterfactual' or pair['human_labeled'] is not False
                    or pair['on_policy'] is not False):
                raise ValueError('Synthetic preference origin is misrepresented')
            if (reward(pair['chosen'], original['ground_truth']) != 1
                    or reward(pair['rejected'], original['ground_truth']) != 0):
                raise ValueError('DPO chosen/rejected correctness failed')
            stats = token_stats(tokenizer, manifest['tokenization'], pair['prompt'], [pair['chosen'], pair['rejected']])
            if stats != pair['token_counts'] or stats['longest_sequence'] > 1024 or stats['unknown_prompt_tokens']:
                raise ValueError('Retokenization/context verification failed')
            maximum = max(maximum, stats['longest_sequence'])
            language_answers[pair['language']] += sum(stats['answers_with_eos'])
        for group in group_rows.values():
            if len(group) != 2 or {row['language'] for row in group} != {'zh', 'en'}:
                raise ValueError('Bilingual problem group was split or duplicated')
            if group[0]['ground_truth'] != group[1]['ground_truth'] or group[0]['metadata'] != group[1]['metadata']:
                raise ValueError('Bilingual math meaning changed')
        if ids & all_ids or groups & all_groups or expressions & all_expressions or questions & all_questions:
            raise ValueError('Cross-split problem/expression/question leakage')
        if len(questions) != len(data) or len(groups) != expected['groups']:
            raise ValueError('Question/group duplicates found')
        all_ids |= ids; all_groups |= groups; all_expressions |= expressions; all_questions |= questions
        languages = dict(Counter(row['language'] for row in data))
        families = dict(Counter(row['metadata']['family'] for row in data))
        if languages != expected['languages'] or families != expected['families'] or set(families) != set(FAMILIES):
            raise ValueError('Stratified coverage differs from manifest')
        if languages['zh'] != languages['en'] or language_answers['zh'] != language_answers['en']:
            raise ValueError('Bilingual row/answer-token balance differs')
        if split == 'train':
            for view, full in (('rlvr', data), ('dpo', preferences)):
                for name, spec in manifest['files'].items():
                    if name.startswith(view + '/train_'):
                        subset = rows(folder / spec['path'])
                        if subset != full[:spec['examples']] or len(subset) % 2:
                            raise ValueError('Nested training version changed')
        split_reports[split] = {'examples': len(data), 'groups': len(groups), 'languages': languages,
                                'families': families, 'max_sequence_tokens': maximum,
                                'zh_answer_token_fraction': language_answers['zh'] / sum(language_answers.values())}
    public_reports = {}
    for name, spec in manifest['files'].items():
        if not name.startswith('external_test/'):
            continue
        data = rows(folder / spec['path'])
        if len(data) != spec['examples'] or len({row['id'] for row in data}) != len(data):
            raise ValueError('External test record count/IDs changed')
        for row in data:
            if row['original_split'] != 'test' or 'test' not in row['source_file']:
                raise ValueError('External test came from a training source')
            if question_hash(row['question'], converter) in exclusions:
                raise ValueError('Known external-test question overlap')
            if converter.convert(row['prompt']) != row['prompt']:
                raise ValueError('External Chinese normalization changed')
            length = len(tokenizer.encode(prefix(row['prompt'], manifest['tokenization']), add_special_tokens=False).ids)
            if length != row['prompt_tokens'] or length + 128 > 1024:
                raise ValueError('Complete public question does not fit the context')
            if reward(row['ground_truth'], row['ground_truth'], row['reward_type']) != 1:
                raise ValueError('Invalid public reference label')
        public_reports[name.split('/', 1)[1]] = {'eligible_examples': len(data), 'official_examples': spec['official_examples'],
                                               'languages': dict(Counter(row['language'] for row in data)),
                                               'difficulty': dict(Counter(row['difficulty'] for row in data))}
    if require_acceptance:
        acceptance = manifest.get('acceptance', {})
        report_path = safe_path(folder, acceptance.get('report', 'acceptance.json'))
        if acceptance.get('status') != 'passed' or sha256(report_path) != acceptance.get('sha256'):
            raise ValueError('Math release acceptance is missing or changed')
        if read_json(report_path).get('data_identity') != manifest['data_identity']:
            raise ValueError('Acceptance belongs to a different release')
    return {'status': 'passed', 'data_identity': manifest['data_identity'],
            'checks': {'file_hashes': True, 'provenance': True, 'retokenization_and_context': True,
                       'independent_exact_arithmetic': True, 'chosen_correct_rejected_incorrect': True,
                       'bilingual_group_isolation': True, 'numeric_expression_split_isolation': True,
                       'heldout_template_isolation': True, 'simplified_chinese_fixed_point': True,
                       'nested_training_versions': True, 'known_project_question_exclusion': True,
                       'official_test_origin_and_no_question_truncation': True},
            'splits': split_reports, 'external_tests': public_reports,
            'limits': ['DPO negatives are programmatically constructed, not sampled from the SFT model or human preferences.',
                       'Public labels are source-provided; mathematical correctness of all public references is not independently proved.',
                       'Exact known-question isolation does not certify semantic or pretraining decontamination.',
                       'No GPU training or model evaluation was started by data preparation.']}
