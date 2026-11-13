"""构建数学题库的RLVR与DPO两个视图，独立验收通过后原子发布。"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .problems import FAMILIES, make_problem
from .verifier import numeric_value, question_hash


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    return {'path': path.name, 'examples': len(rows), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def code_identity():
    folder = Path(__file__).parent
    return canonical_hash({path.name: sha256(path) for path in sorted(folder.glob('*.py'))})


def manifest_identity(manifest):
    return canonical_hash({key: value for key, value in manifest.items()
                           if key not in {'acceptance', 'data_identity', 'status'}})


def prefix(prompt, protocol):
    return protocol['empty_system_prefix_template'].format(prompt=prompt, system='')


def token_stats(tokenizer, protocol, prompt, answers):
    encoded = tokenizer.encode(prefix(prompt, protocol), add_special_tokens=False).ids
    counts = [len(tokenizer.encode(answer, add_special_tokens=False).ids) + 1 for answer in answers]
    return {'prompt': len(encoded), 'answers_with_eos': counts,
            'longest_sequence': len(encoded) + max(counts), 'unknown_prompt_tokens': encoded.count(1)}


def build_groups(count, split, rng, converter, exclusions, used_expressions):
    if count % 2:
        raise ValueError('Bilingual record counts must be even')
    groups = []
    for index in range(count // 2):
        family = FAMILIES[index % len(FAMILIES)]
        template = rng.choice((0, 1)) if split in {'train', 'test_id'} else 2 if split == 'validation' else rng.choice((3, 4))
        for _ in range(20000):
            problem = make_problem(family, rng, template)
            if problem['expression_key'] in used_expressions:
                continue
            if any(question_hash(text, converter) in exclusions for text in problem['prompts'].values()):
                continue
            used_expressions.add(problem['expression_key'])
            problem['group_id'] = 'arith_' + problem['expression_key'][:24]
            problem['split'] = split
            groups.append(problem)
            break
        else:
            raise ValueError('Cannot produce enough independent bilingual problems')
    if split == 'train':
        # 小规模版本覆盖所有题型，并优先包含数值较小的题，而非任意截取文件前几行。
        buckets = {family: sorted([row for row in groups if row['family'] == family],
                                  key=lambda row: (row['max_operand'], row['expression_key'])) for family in FAMILIES}
        groups = [buckets[family][i] for i in range(max(map(len, buckets.values())))
                  for family in FAMILIES if i < len(buckets[family])]
    return groups


def views(groups, converter, tokenizer, protocol):
    rlvr, dpo = [], []
    instructions = {'zh': '只输出最终答案的数字，不要输出单位或解释。',
                    'en': 'Return only the final numeric answer, without units or explanation.'}
    for group in groups:
        for language in ('zh', 'en'):
            question = converter.convert(group['prompts'][language])
            prompt = question + '\n' + instructions[language]
            identifier = group['group_id'] + '_' + language
            meta = {key: group[key] for key in ('family', 'difficulty', 'template_id', 'expression', 'expression_key', 'split')}
            stats = token_stats(tokenizer, protocol, prompt, [group['ground_truth'], group['rejected_answer']])
            if stats['longest_sequence'] > 1024 or stats['unknown_prompt_tokens']:
                raise ValueError('Generated problem violates the frozen tokenizer/context contract')
            common = {'id': identifier, 'group_id': group['group_id'], 'language': language,
                      'question': question, 'prompt': prompt, 'metadata': meta, 'token_counts': stats,
                      'source': 'programmatic_bilingual_arithmetic_v1'}
            rlvr.append({**common, 'ground_truth': group['ground_truth'], 'reward_type': 'numeric_exact'})
            dpo.append({**common, 'chosen': group['ground_truth'], 'rejected': group['rejected_answer'],
                        'pair_origin': 'programmatic_counterfactual', 'human_labeled': False, 'on_policy': False})
    return rlvr, dpo


def _options(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_options(item))
        return result
    if isinstance(value, dict) and 'aoVal' in value and 'content' in value:
        return [(str(value['aoVal']).strip().upper(), str(value['content']).strip())]
    return []


def public_tests(cache, lock, converter, tokenizer, protocol, exclusions):
    prepared, rejected, raw_counts = {}, [], {}
    seen_questions = set()
    for source in lock['sources']:
        path = cache / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError('Pinned public source changed')
        if source['role'] != 'external_test_only':
            continue
        gsm = source['repository'] == 'openai/grade-school-math'
        language = 'en' if gsm or source['upstream_path'].startswith('en_') else 'zh'
        name = 'gsm8k_en' if gsm else 'tal_' + language
        rows = []
        raw_count = 0
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream):
                row = json.loads(line)
                raw_count += 1
                identifier = name + '_' + str(number)
                reason = None
                if gsm:
                    question = row['question'].strip()
                    answer = row['answer'].rsplit('####', 1)[-1].strip()
                    kind = 'numeric_exact'
                    if numeric_value(answer) is None:
                        reason = 'invalid_official_numeric_reference'
                    instruction = 'Return only the final numeric answer, without units or explanation.'
                    difficulty = 'official_grade_school_multistep'
                else:
                    question = row['problem'].strip()
                    options = _options(row.get('answer_option_list'))
                    labels = [label for label, _ in options]
                    answer = str(row['answer_value']).strip().upper()
                    kind = 'choice_exact'
                    if (not re.fullmatch('[A-H]', answer) or answer not in labels or len(labels) != len(set(labels))
                            or any(not re.fullmatch('[A-H]', label) for label in labels)):
                        reason = 'invalid_official_single_choice_reference_or_options'
                    question += '\n' + '\n'.join(label + '. ' + text for label, text in options)
                    instruction = ('只输出正确选项的字母，不要输出解释。' if language == 'zh'
                                   else 'Return only the letter of the correct option, without explanation.')
                    difficulty = str(row.get('difficulty', 'unknown'))
                question = converter.convert(question)
                prompt = question + '\n' + instruction
                digest = question_hash(question, converter)
                bare_digest = question_hash(row.get('problem', row.get('question', '')), converter)
                if digest in exclusions or bare_digest in exclusions:
                    reason = 'known_project_question_overlap'
                if digest in seen_questions:
                    reason = 'duplicate_official_question'
                prompt_tokens = len(tokenizer.encode(prefix(prompt, protocol), add_special_tokens=False).ids)
                if prompt_tokens + 128 > 1024:
                    reason = 'complete_question_plus_128_generation_tokens_exceeds_1024'
                if reason:
                    rejected.append({'id': identifier, 'source': name, 'reason': reason})
                    continue
                seen_questions.add(digest)
                rows.append({'id': identifier, 'group_id': name + '_' + digest[:24], 'language': language,
                             'question': question, 'prompt': prompt, 'ground_truth': answer, 'reward_type': kind,
                             'source': source['repository'], 'source_revision': source['revision'],
                             'source_file': source['upstream_path'], 'source_row': number, 'original_split': 'test',
                             'difficulty': difficulty, 'prompt_tokens': prompt_tokens,
                             'reference_status': 'official_source_label_not_independently_solved'})
        prepared[name] = rows
        raw_counts[name] = raw_count
    return prepared, rejected, raw_counts


def build(config_path):
    from opencc import OpenCC
    from tokenizers import Tokenizer
    config_path = Path(config_path).resolve()
    settings = read_json(config_path)
    root = (config_path.parent / settings['project_root']).resolve()
    destination = (root / settings['output_dir']).resolve()
    if not destination.is_relative_to(root / 'artifacts/posttrain'):
        raise ValueError('Math publication must stay in project artifacts/posttrain')
    if destination.exists():
        raise FileExistsError('A published release may not be overwritten')
    staging = destination.with_name(destination.name + '.building')
    staging.mkdir(parents=True, exist_ok=False)
    converter = OpenCC('t2s')
    cache = root / settings['source_cache']
    lock_path = cache / 'sources.lock.json'
    lock = read_json(lock_path)
    context_path = cache / 'project_exclusions.json'
    context = read_json(context_path)
    exclusion_hashes = set(context['question_hashes'])
    tokenizer_path = root / settings['tokenizer']
    if sha256(tokenizer_path) != context['tokenizer']['sha256']:
        raise ValueError('Existing 300M tokenizer differs from the source corpus')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    protocol = context['tokenization']
    used = set()
    splits, files = {}, {}
    for index, (split, count) in enumerate(settings['counts'].items()):
        groups = build_groups(count, split, random.Random(settings['seed'] + 1009 * index),
                              converter, exclusion_hashes, used)
        rlvr, dpo = views(groups, converter, tokenizer, protocol)
        splits[split] = {'examples': count, 'groups': len(groups),
                         'languages': dict(Counter(row['language'] for row in rlvr)),
                         'families': dict(Counter(row['metadata']['family'] for row in rlvr))}
        for view, rows in (('rlvr', rlvr), ('dpo', dpo)):
            path = staging / view / (split + '.jsonl')
            spec = write_rows(path, rows)
            spec['path'] = path.relative_to(staging).as_posix()
            files[view + '/' + split] = spec
            if split == 'train':
                for size in settings['nested_training_examples']:
                    if size > len(rows) or size % 2:
                        raise ValueError('Nested versions must retain whole bilingual groups')
                    subpath = staging / view / f'train_{size}.jsonl'
                    sub = write_rows(subpath, rows[:size])
                    sub['path'] = subpath.relative_to(staging).as_posix()
                    files[view + '/train_' + str(size)] = sub
        print(json.dumps({'phase': 'generated', 'split': split, 'examples': count}, ensure_ascii=False), flush=True)
    external, rejected, raw_counts = public_tests(cache, lock, converter, tokenizer, protocol, exclusion_hashes)
    for name, rows in external.items():
        path = staging / 'external_test' / (name + '.jsonl')
        spec = write_rows(path, rows)
        spec.update(path=path.relative_to(staging).as_posix(), official_examples=raw_counts[name])
        files['external_test/' + name] = spec
    provenance = staging / 'provenance'
    provenance.mkdir()
    shutil.copy2(lock_path, provenance / 'sources.lock.json')
    shutil.copy2(context_path, provenance / 'project_exclusions.json')
    shutil.copy2(cache / 'tal_hf_dataset_card.md', provenance / 'tal_hf_dataset_card.md')
    shutil.copy2(cache / 'openai__grade-school-math/LICENSE', provenance / 'gsm8k_LICENSE')
    shutil.copy2(config_path, provenance / 'build_config.json')
    write_json(provenance / 'external_exclusions.json', rejected)
    shutil.copy2(tokenizer_path, staging / 'tokenizer.json')
    provenance_files = {path.name: sha256(path) for path in sorted(provenance.iterdir())}
    manifest = {'schema_version': 1, 'kind': 'bilingual_arithmetic_alignment', 'status': 'awaiting_independent_audit',
                'fixture': settings.get('fixture', False),
                'seed': settings['seed'], 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'synthetic_splits': splits, 'files': files, 'provenance_files': provenance_files,
                'tokenizer': {**context['tokenizer'], 'path': 'tokenizer.json'}, 'tokenization': protocol,
                'max_sequence_tokens': 1024, 'max_external_generation_tokens': 128,
                'data_code_sha256': code_identity(), 'training_policy': {'epochs': 1, 'stop_epochs': None},
                'dpo_pair_origin': 'programmatic_counterfactual_not_human_or_on_policy',
                'public_training_records_are_reserve_only': True,
                'evaluation': {'in_domain': 'rlvr/test_id.jsonl', 'unseen_templates': 'rlvr/test_template.jsonl',
                               'external_sources_scored_separately': True, 'test_used_for_tuning': False,
                               'official_raw_tests_retained_in_source_cache': True,
                               'known_project_exclusion_hashes': len(exclusion_hashes),
                               'decontamination_scope': 'normalized exact known project questions; not a guarantee of semantic or pretraining decontamination'},
                'training_started': False}
    manifest['data_identity'] = manifest_identity(manifest)
    write_json(staging / 'manifest.json', manifest)
    from .audit import audit
    report = audit(staging / 'manifest.json', require_acceptance=False)
    write_json(staging / 'acceptance.json', report)
    manifest.update(status='published', acceptance={'status': 'passed', 'report': 'acceptance.json',
                                                   'sha256': sha256(staging / 'acceptance.json')})
    write_json(staging / 'manifest.json', manifest)
    # 验收后才发布可见目录和latest指针，不覆盖任何旧语料或模型。
    staging.rename(destination)
    pointer = {'schema_version': 1, 'kind': manifest['kind'], 'manifest': str(destination / 'manifest.json'),
               'manifest_sha256': sha256(destination / 'manifest.json'), 'data_identity': manifest['data_identity'],
               'training_started': False}
    write_json(root / 'artifacts/posttrain/latest_math_alignment.json', pointer)
    return {**pointer, 'synthetic_splits': splits,
            'external_tests': {name: {'eligible': len(rows), 'official': raw_counts[name]} for name, rows in external.items()},
            'external_exclusions': len(rejected), 'acceptance': report}
