"""对未发布候选作第二轮保守筛选；只改任务标签，不改文本或Token。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import shutil
import time

import prepare_sft_data as base
from prepare_sft_expansion import CN_TASKS, TEXT_FIELDS, constraint_reason, row_stream

VERSION = 'strict_task_and_explicit_constraints_v1'
OTHER = 'short_qa_or_other'
WORD = r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*"
COMPARISON = r'at least|at most|exactly|no more than|less than|fewer than'
FOREIGN_SCRIPT = re.compile('[\u0400-\u052f\u0600-\u06ff\u0e00-\u0e7f\u3040-\u30ff\uac00-\ud7af]')


def violates(count, comparison, limit):
    if comparison == 'at least': return count < limit
    if comparison in {'at most', 'no more than'}: return count > limit
    if comparison in {'less than', 'fewer than'}: return count >= limit
    return count != limit


def explicit_constraints(prompt, response, language):
    """计数采用公开固定规则；不能解析的自然语言约束不冒充已验证。"""
    reason = constraint_reason(prompt, response, language)
    if reason: return reason
    if re.search(r'(?:all lowercase|all lowercase letters|no capital letters)', prompt, re.I):
        if re.search('[A-Z]', response): return 'required_lowercase_violated'
    quoted = r"(?:'([^']{1,60})'|\"([^\"]{1,60})\"|\[([^\]\n]{1,60})\])"
    pattern = r'\bword\s+' + quoted + r'\s+(?:should|must)\s+(?:appear|occur)\s+(' + COMPARISON + r')\s+(\d+)\s+times'
    for match in re.finditer(pattern, prompt, re.I):
        word = next(value for value in match.groups()[:3] if value is not None)
        count = len(re.findall(r'(?<!\w)' + re.escape(word) + r'(?!\w)', response, re.I))
        if violates(count, match.group(4).lower(), int(match.group(5))):
            return 'required_keyword_frequency_violated'
    for match in re.finditer(r'\b(' + COMPARISON + r')\s+(\d+)\s+words\b', prompt, re.I):
        if violates(len(re.findall(WORD, response)), match.group(1).lower(), int(match.group(2))):
            return 'explicit_word_budget_violated'
    for match in re.finditer(r'\b(' + COMPARISON + r')\s+(\d+)\s+bullet\s+points\b', prompt, re.I):
        count = len(re.findall(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+', response))
        if violates(count, match.group(1).lower(), int(match.group(2))):
            return 'explicit_bullet_budget_violated'
    for match in re.finditer(r'\b(' + COMPARISON + r')\s+(\d+)\s+placeholders\b', prompt, re.I):
        count = len(re.findall(r'\[[^\]\n]+\]', response))
        if violates(count, match.group(1).lower(), int(match.group(2))):
            return 'explicit_placeholder_budget_violated'
    for match in re.finditer(r'include keywords?\s*\[([^\]]+)\]', prompt, re.I):
        words = [word.strip() for word in match.group(1).split(',')]
        if any(not re.search(r'(?<!\w)' + re.escape(word) + r'(?!\w)', response, re.I) for word in words):
            return 'required_keyword_missing'
    if re.search(r'postscript starting with\s+p\.s\.', prompt, re.I):
        if not re.search(r'(?im)^\s*p\.s\.', response): return 'required_postscript_missing'
    # “每个步骤的说明15-20字”只计算冒号后的说明，排除编号和标题。
    match = re.search(r'每(?:个|一)(?:步骤|项).{0,12}(\d+)\s*[-–到至]\s*(\d+)\s*字', prompt)
    if match:
        descriptions = [re.split('[:：]', line, maxsplit=1)[-1] for line in response.splitlines()
                        if re.match(r'^\s*\d+[.、)]\s*', line)]
        if not descriptions or any(not int(match.group(1)) <= sum(c.isalnum() for c in text) <= int(match.group(2))
                                   for text in descriptions):
            return 'per_item_chinese_character_budget_violated'
    return None


def clarification(row):
    text = row['response']
    return bool(re.search(
        r'(?:请|需要).{0,18}(?:提供|补充|发给).{0,25}(?:原文|文章|文本|材料|背景|问题|内容)|'
        r'(?:没有|未|尚未).{0,8}(?:提供|收到).{0,12}(?:文章|文本|材料)|'
        r'(?:please|feel free to|could you|can you).{0,35}(?:provide|share|send).{0,45}'
        r'(?:text|article|lecture|podcast|material|context)|'
        r'(?:已|已经|我已).{0,8}(?:阅读|理解|了解).{0,70}(?:请问|告诉我|问题)', text, re.I | re.S))


def strict_task(row):
    """来源标签优先；一般问答不能靠英文子串或“情感”等名词变为操作任务。"""
    if clarification(row): return OTHER
    subset = row['source']['source_subset']
    if subset in CN_TASKS: return CN_TASKS[subset]
    known = {'smollm-rewrite-30k': 'rewriting', 'smol-summarize-20k': 'summarization',
             'smol-contraints': 'format_following', 'information_extraction': 'extraction',
             'closed_qa': 'grounded_qa'}
    if subset in known: return known[subset]
    text = row['system'] + '\n' + row['prompt']
    material = bool(row['input_context']) or bool(re.search(r'以下|下列|如下|这句话|给定|following|below', text, re.I))
    if re.search(r'\b(?:rewrite|rephrase|paraphrase|proofread|revise)\b|改写|重写|润色', text, re.I):
        return 'rewriting'
    if material and re.search(r'\b(?:summarize|summarise)\b|总结|概括|摘要', text, re.I):
        return 'summarization'
    if material and re.search(r'\bextract\b|(?:提取|抽取|找出).{0,16}(?:以下|下列|句子|文本)|'
                              r'(?:从|在).{0,18}(?:文本|句子|文章).{0,10}(?:提取|抽取|找出)', text, re.I):
        return 'extraction'
    if material and re.search(r'\b(?:classify|categorize|label)\b|(?:将|把).{0,20}(?:分类|归类)|'
                              r'分类(?:以下|下列)|判断.{0,12}(?:正面|负面|中性)', text, re.I):
        return 'classification'
    if re.search(r'(?:返回|输出|使用|用|以).{0,16}(?:JSON|大写|小写|格式)|'
                 r'\b(?:return|output|respond|write|use)\b.{0,30}\b(?:json|lowercase|uppercase)\b|'
                 r'\b(?:exactly|at least|at most)\s+\d+\s+(?:words|bullet points)\b', text, re.I):
        return 'format_following'
    if row['input_context']: return 'grounded_qa'
    return OTHER


def quarantine_reason(row, decisions):
    decision = decisions.get(row['id'])
    if decision and decision['decision'] == 'exclude': return 'documented_content_review_exclusion'
    family = row['source']['reserve_family']
    # 本版本保守收录已经逐条审过的OASST；不把抽审错误比例推广到整个来源。
    if family == 'oasst2' and not (decision and decision['decision'] == 'accept'):
        return 'oasst_not_individually_reviewed_for_this_release'
    prompt = row['system'] + '\n' + row['prompt']
    response = row['response']
    if row['language'] == 'zh' and FOREIGN_SCRIPT.search(prompt + '\n' + response):
        return 'outside_bilingual_script_curriculum'
    subset = row['source']['source_subset']
    if subset == 'brainstorming.parquet' and re.search(
        r'生成_assistant|generate\s+Assistant|\n\n(?:当然可以|这个想法非常有潜力)|'
        r'\b(?:assistant|user)\s*:', row['prompt'], re.I):
        return 'user_prompt_contains_assistant_role_or_answer'
    if re.search(r'https?://', row['prompt']) and not row['input_context']:
        return 'external_material_not_fetched_or_verified'
    if re.search(r'\b(?:sudo|apt-key|pip install|apt install)\b', response) and re.search(
        r'\b(?:build|create|implement|bot|install|configure)\b|搭建|开发|安装', prompt, re.I):
        return 'technical_execution_outside_initial_curriculum'
    if re.search(r'\b(?:orthodont|dentist|dental|pediatric)\w*|正畸|牙医|牙科', prompt + response, re.I):
        return 'specialist_health_claims_not_verified'
    if not row['input_context'] and re.search(
        r'\b(?:recommend|suggest)\b.{0,55}\b(?:websites?|blogs?|software|tools?)\b|'
        r'推荐.{0,24}(?:网站|博客|软件|工具)', prompt, re.I | re.S):
        return 'current_recommendations_not_verified'
    return explicit_constraints(prompt, response, row['language'])


def curate(directory, review_path, work, min_free=18*1024**3, max_bytes=6*1024**3):
    directory = Path(directory); work = Path(work); review_path = Path(review_path)
    pool_path = directory/'pool_manifest.json'; candidates = directory/'candidates.jsonl'
    original_manifest = directory/'pool_manifest.stage1.json'
    original_candidates = directory/'candidates.stage1.jsonl'
    report_path = directory/'pool_curation.json'
    if any(path.exists() for path in (original_manifest, original_candidates, report_path)):
        raise ValueError('Refusing to re-curate or overwrite archived candidates')
    if (directory/'release.json').exists(): raise ValueError('Published data cannot be curated in place')
    pool = json.loads(pool_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    parent_sha = base.sha256(candidates)
    if parent_sha != pool['candidates_sha256'] or parent_sha != review['pool_sha256']:
        raise ValueError('Review and source pool identities differ')
    samples = list(row_stream(directory/'review_sample.jsonl'))
    decisions = {row['id']: row for row in review['reviewed']}
    if (set(decisions) != {row['id'] for row in samples} or len(review['reviewed']) != len(samples)
        or review.get('test_content_used') is not False):
        raise ValueError('Documented review incomplete or test was inspected')
    temporary = directory/'candidates.curating.jsonl'
    rejected = Counter(); tasks = Counter(); languages = Counter(); families = Counter(); changes = Counter()
    started = time.monotonic(); seen = kept = 0
    with temporary.open('x', encoding='utf-8') as stream, (work/'curation_rejections.jsonl').open('x', encoding='utf-8') as exclusions:
        for row in row_stream(candidates):
            seen += 1
            reason = quarantine_reason(row, decisions)
            if reason:
                rejected[reason] += 1
                exclusions.write(json.dumps({'id': row['id'], 'reason': reason})+'\n')
            else:
                task = strict_task(row)
                if task != row['task']: changes[row['task']+' -> '+task] += 1
                row['task'] = task
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
                kept += 1; tasks[task] += 1; languages[row['language']] += 1; families[row['source']['reserve_family']] += 1
            if seen % 20000 == 0:
                if shutil.disk_usage(directory).free < min_free: raise OSError('Minimum disk reserve reached')
                total = sum(p.stat().st_size for folder in (directory, work) for p in folder.rglob('*') if p.is_file())
                if total > max_bytes: raise OSError('Bounded output budget reached')
                print(json.dumps({'phase': 'curating', 'seen': seen, 'kept': kept, 'rejected': sum(rejected.values())}), flush=True)
        stream.flush(); os.fsync(stream.fileno())
    assert seen == pool['candidate_count'] and seen == kept + sum(rejected.values())
    report = {'version': VERSION, 'parent_pool_sha256': parent_sha, 'parent_candidate_count': seen,
              'candidate_count': kept, 'candidate_sha256': base.sha256(temporary), 'rejections': dict(rejected),
              'task_relabels': dict(changes), 'languages': dict(languages), 'tasks': dict(tasks), 'families': dict(families),
              'script_sha256': base.sha256(Path(__file__)), 'text_and_tokens_modified': False,
              'oasst_admission': 'Only documented individually accepted review examples; remainder kept in raw reserve',
              'word_count_rule': WORD, 'per_item_character_rule': 'Unicode letters and digits after first colon; exclude punctuation',
              'semantic_gold': False, 'test_content_used': False, 'elapsed_seconds': time.monotonic()-started}
    # 原始文件同盘重命名保留完整证据；变更仅限尚未发布的候选池。
    shutil.copy2(pool_path, original_manifest)
    shutil.copy2(review_path, work/'review_decisions.stage1.json')
    candidates.rename(original_candidates); temporary.rename(candidates)
    pool.update(candidate_count=kept, languages=dict(languages), tasks=dict(tasks), families=dict(families),
                candidates_sha256=report['candidate_sha256'], curation=report,
                status='curated_with_documented_review_pending_publication')
    review.update(parent_pool_sha256=parent_sha, pool_sha256=report['candidate_sha256'], pool_curation=report,
                  review_text_scope='Original 64 texts unchanged; only deterministic task annotations and conservative subset admission updated')
    base.atomic_json(report_path, report); base.atomic_json(pool_path, pool)
    base.atomic_json(work/'review_decisions.json', review)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'review', 'work-dir'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[-2:]))
    if hasattr(os, 'nice'): os.nice(10)
    print(json.dumps(curate(args.data_dir, args.review, args.work_dir), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
