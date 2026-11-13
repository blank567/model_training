"""独立 SFT 数据启发式：它只筛明显问题，不证明语义正确或完整去污染。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import re
import unicodedata


def normalize(text: str) -> str:
    """统一 Unicode 和行内空白；保留段落，繁转简由外层官方 OpenCC 完成。"""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fingerprint(text: str) -> str:
    """用于精确/近重复键；忽略大小写、标点与空白，不把空串作为键。"""
    return re.sub(r"[\W_]+", "", normalize(text).casefold())


def quality_reason(prompt: str, answer: str, language: str, source: str = "") -> str | None:
    """返回首个拒收原因；None 只表示未触发规则，绝不等于人工金标。"""
    prompt, answer = normalize(prompt), normalize(answer)
    if language not in {"en", "zh"}:
        return "unsupported_language"
    if not prompt or not answer:
        return "empty_prompt_or_answer"
    if not fingerprint(answer):
        return "noninformative_answer"
    text = prompt + "\n" + answer
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]|<\|[^\n]{1,50}?\|>", text):
        return "control_or_role_marker"
    if answer[-1] in ",，:：" or answer.endswith(("...", "…")):
        return "unfinished_answer"
    if "```" in text or re.search(r"(?m)^\s*(?:def |class |import |#include|SELECT\s+.*FROM\s)", text) or re.search(r"C#|C\+\+|\b(?:python|javascript|typescript|nodejs|dotnet|programming|compiler|SQL|API)\b|编程|程序代码|写代码|程序运行",prompt,re.I):
        return "code_task"
    if re.search(r"<think>|chain[- ]of[- ]thought|let['’]s think step by step|逐步推理|思维链", text, re.I):
        return "reasoning_trace"
    if re.search(r"\b(?:diagnos\w*|clinical|patients?|medications?|injur\w*|symptoms?|illness|disease|treatment|therap\w*|surger\w*|cancer|diabetes|strain|lawsuit|attorney|legal advice)\b|诊断|患者|药物|医疗|受伤|症状|疾病|治疗|处方|用药|法律意见|诉讼", text, re.I):
        return "specialist_medical_or_legal"
    if re.search(r"\b(?:1\d{3}|20\d{2})\b|(?:1\d{3}|20\d{2})年", text):
        return "dated_fact_task"
    if re.search(r"\b(?:calculate|compute|solve|equation|integral|derivative|fractions?|median|variance|probability|correlation|regression|factoriz\w*|standard deviation)\b|计算下|求解|方程|微积分|数学证明|因式|多项式|表达式|代数|求值|倍数|均值|中位数|众数|概率|组合数|排列数|算式|平方|立方", prompt, re.I):
        return "math_task"
    if re.search(r"\d\s*[/⁄]\s*\d|\b[a-z]\s*[²³^=+*/]|\d\s*[+*=^]|[=+*/]\s*\d", text, re.I):
        return "math_expression_task"
    # 长叙述停在普通汉字/字母上是截断风险；短标签和有项目符号的列表不因此误拒。
    terminal=answer.rstrip('"\'”’）)]】 ')
    last_line=answer.splitlines()[-1].strip()
    bullet=bool(re.match(r"(?:[-*•]|\d+[.)、])\s*",last_line))
    if not bullet and terminal and terminal[-1] not in '.!?。！？;；':
        if language=='zh' and len(re.findall(r'[\u4e00-\u9fff]',answer))>=40:
            return "possibly_truncated_prose"
        if language=='en' and len(re.findall(r'[A-Za-z]',answer))>=120 and re.search(r'[A-Za-z]$',terminal):
            return "possibly_truncated_prose"
    material_marker=re.search(r"(?:此|以下|给定|这段|上述|这个)(?:文本|句子|陈述|文章|段落|列表|内容|信息|词语)|从(?:文章|文本)中(?:提取|找出)|(?:this|the following|the given)\s+(?:text|sentence|statement|article|passage|paragraph|list)",prompt,re.I)
    needs_material=bool(material_marker or re.search(r"(?:总结|概括|摘要|归纳).{0,18}(?:文章|文本|段落|原文)|(?:文章|文本|段落).{0,18}(?:总结|概括|摘要)|summariz\w*\s+(?:the|this|following)\s+(?:article|text|passage|paragraph|document)",prompt,re.I))
    clarification=bool(re.search(r"请(?:您|你)?提供|需要(?:知道|您|你).{0,12}(?:文章|原文|文本|内容)|(?:please|could you)\s+provide|need\s+(?:the|your)\s+(?:text|article|sentence)",answer,re.I))
    if needs_material and '\n' not in prompt and not clarification:
        inline=re.split(r'[:：]',prompt,maxsplit=1)
        quoted=re.findall(r'[“"]([^”"]{6,})[”"]',prompt)
        has_inline=(len(inline)==2 and len(inline[1].strip())>=12) or bool(quoted)
        if material_marker and re.search(r'[。.!?]',prompt[:material_marker.start()]) and len(prompt[:material_marker.start()])>=8:
            has_inline=True
        if not has_inline:
            return "missing_source_material"
    if language=='zh' and 0<len(re.findall(r'[\u4e00-\u9fff]',prompt))<8:
        if not re.search(r'[?？]|请|写|列|解释|说|翻译|改写|总结|提取|给|生成|将|把|问|答|判断|比较|分类|返回|保持|什么|为何|如何|怎么|是否|你好|您好|谢谢|再见|对不起|早上|晚上|晚安',prompt):
            return "underspecified_short_prompt"
    if re.search(r"\b(?:latest|current weather|today['’]s|real[- ]time)\b|最新|实时|今日天气|今天.*天气", prompt, re.I):
        return "time_sensitive_task"
    if re.search(r"\bI\s+(?:have\s+|already\s+|just\s+)(?:successfully\s+)?(?:sent|booked|deleted|purchased|accessed|downloaded)\b|\bI['’]ve\s+(?:sent|booked|deleted|purchased|accessed|downloaded)\b|我(?:已经|已)(?:为你|帮你)?(?:发送|预订|删除|购买|访问|下载|联网|查询了)", answer, re.I):
        return "unsupported_execution_claim"
    if re.search(r"\bas an (?:AI|artificial intelligence|language model)\b|作为(?:一个)?(?:AI|人工智能|语言模型)", answer, re.I):
        return "identity_boilerplate"
    compact = re.sub(r"\s+", "", answer)
    if len(compact) >= 20 and max(Counter(compact).values()) / len(compact) > .45:
        return "character_repetition"
    lines = [line for line in answer.splitlines() if line.strip()]
    if len(lines) >= 3 and len(set(lines)) / len(lines) < .6:
        return "line_repetition"
    if re.search(r"(.{4,80})\1{3,}", compact):
        return "phrase_repetition"
    if fingerprint(prompt) == fingerprint(answer):
        return "prompt_echo"
    if language == "en":
        numbers = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        limit = re.search(r"(?:up to|at most|no more than|maximum of)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences", prompt, re.I)
        if limit:
            value = limit.group(1).casefold()
            maximum = int(value) if value.isdigit() else numbers[value]
            sentences = [part for part in re.split(r"[.!?]+(?:\s+|$)", answer) if part.strip()]
            if len(sentences) > maximum:
                return "sentence_limit_violation"
        if re.search(r"(?:without|not|avoid|no)\b.{0,100}second.{0,25}third.{0,25}person pronouns", prompt, re.I | re.S):
            if re.search(r"\b(?:you|your|yours|yourself|he|him|his|she|her|hers|they|them|their|theirs|it|its)\b", answer, re.I):
                return "pronoun_constraint_violation"
        if re.search(r"(?:all|entirely|only)\s+(?:in\s+)?lowercase", prompt, re.I) and re.search(r"[A-Z]", answer):
            return "lowercase_constraint_violation"
    return None


def _shingles(text: str) -> set[int]:
    # 首尾都取，单字段最多 4096 字符；候选索引是近似检查，有明确漏检边界。
    text = fingerprint(text)
    if len(text) < 40:
        return set()
    if len(text) > 4096:
        text = text[:2048] + text[-2048:]
    return {int.from_bytes(hashlib.blake2b(text[i:i+5].encode(), digest_size=8).digest(), "big")
            for i in range(len(text)-4)}


def group_candidates(records: list[dict], threshold: float = .80, max_bucket: int = 128) -> list[str]:
    """同问题/材料/来源组及字符近重复归并；结果与输入顺序一致。

    字段为 prompt、answer、input_context、source_group_id、id。source_group_id
    由调用方提供带来源命名空间的对话/材料 ID。桶满后仅做有界候选检查；
    不承诺所有字符近重复、改写或跨语言翻译都被识别。
    """
    if not 0 < threshold <= 1 or max_bucket < 1:
        raise ValueError("invalid grouping limits")
    parents = list(range(len(records)))
    identities = [hashlib.sha256((str(row.get("id", ""))+"\0"+normalize(row.get("prompt", ""))+"\0"+normalize(row.get("answer", ""))).encode()).hexdigest() for row in records]
    order = sorted(range(len(records)), key=lambda i: (identities[i], i))
    exact: dict[tuple[str, str], int] = {}
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    seen_sets: dict[tuple[str, int], set[int]] = {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left,right)] = min(left,right)

    for index in order:
        row = records[index]
        keys = [("prompt", fingerprint(row.get("prompt", ""))),
                ("input_context", fingerprint(row.get("input_context") or "")),
                ("source_group_id", str(row.get("source_group_id") or "").strip())]
        for field, value in keys:
            if value:
                key = (field,value)
                if key in exact:
                    union(index,exact[key])
                else:
                    exact[key] = index
        for field in ("prompt", "input_context"):
            signature = _shingles(row.get(field) or "")
            if not signature:
                continue
            sketches = heapq.nsmallest(8, signature)
            candidates = set()
            for sketch in sketches:
                candidates.update(buckets[(field,sketch)])
            for other in sorted(candidates, key=lambda i: identities[i])[:max_bucket]:
                previous = seen_sets[(field,other)]
                if min(len(signature),len(previous)) / max(len(signature),len(previous)) < threshold:
                    continue
                intersection = len(signature & previous)
                if intersection / (len(signature)+len(previous)-intersection) >= threshold:
                    union(index,other)
            seen_sets[(field,index)] = signature
            for sketch in sketches:
                bucket = buckets[(field,sketch)]
                if len(bucket) < max_bucket:
                    bucket.append(index)
    members: dict[int,list[str]] = defaultdict(list)
    for index in order:
        members[find(index)].append(identities[index])
    group_ids = {root:"group_"+hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest() for root,values in members.items()}
    return [group_ids[find(index)] for index in range(len(records))]
