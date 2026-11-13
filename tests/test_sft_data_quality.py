"""对容易造成坏标签与跨分区泄漏的边界进行独立小样例验证。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("sft_data_quality", Path(__file__).parents[1]/"scripts/sft_data_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_normalize_keeps_paragraphs_and_nfkc():
    assert quality.normalize("  Ａ\t B\r\n\r\n\r\n中\u3000文  ") == "A B\n\n中 文"


@pytest.mark.parametrize("answer", ["positive", "True", "否", "A", "42"])
def test_short_classification_is_allowed(answer):
    assert quality.quality_reason("Return the label for this example.", answer, "en") is None


@pytest.mark.parametrize("prompt,answer,reason", [
    ("问题", "", "empty_prompt_or_answer"),
    ("问题", "未写完，", "unfinished_answer"),
    ("Send the email", "I have sent the email to your client.", "unsupported_execution_claim"),
    ("发邮件", "我已经帮你发送邮件。", "unsupported_execution_claim"),
    ("Question", "Done\nDone\nDone\nDone", "line_repetition"),
    ("Write Python", "```python\nprint(1)\n```", "code_task"),
    ("Explain the event in 1981", "A description.", "dated_fact_task"),
    ("Summarize in up to three sentences.", "One. Two. Three. Four.", "sentence_limit_violation"),
    ("Write without using second or third person pronouns.", "They will meet tomorrow.", "pronoun_constraint_violation"),
    ("Use all lowercase letters.", "Hello world.", "lowercase_constraint_violation"),
])
def test_clear_rejections(prompt,answer,reason):
    assert quality.quality_reason(prompt,answer,"en") == reason


def test_negative_execution_claim_does_not_match():
    assert quality.quality_reason("Can you send this?", "I have not sent it.","en") is None


def test_group_exact_prompt_context_and_source_without_empty_collision():
    rows = [
        {"id":"a","prompt":"Explain this?","answer":"First answer"},
        {"id":"b","prompt":"Explain this!","answer":"Different answer"},
        {"id":"c","prompt":"Question one","answer":"A","input_context":"Shared factual passage."},
        {"id":"d","prompt":"Question two","answer":"B","input_context":"Shared factual passage."},
        {"id":"e","prompt":"Different one","answer":"C","source_group_id":"repo:tree123"},
        {"id":"f","prompt":"Different two","answer":"D","source_group_id":"repo:tree123"},
        {"id":"g","prompt":"Unique prompt seven","answer":"Yes"},
        {"id":"h","prompt":"Unique prompt eight","answer":"Yes"},
    ]
    groups=quality.group_candidates(rows)
    assert groups[0]==groups[1]
    assert groups[2]==groups[3]
    assert groups[4]==groups[5]
    assert len({groups[0],groups[2],groups[4],groups[6],groups[7]}) == 5


def test_near_duplicate_and_order_independent_components():
    base="Please summarize the following ordinary project update while preserving its meaning and including the most important action items."
    rows=[{"id":"a","prompt":base,"answer":"Short summary."},
          {"id":"b","prompt":base.replace("ordinary", "routine"),"answer":"Other summary."},
          {"id":"c","prompt":"Explain photosynthesis in one short sentence.","answer":"Plants use light."}]
    groups=quality.group_candidates(rows)
    assert groups[0]==groups[1]
    assert groups[0]!=groups[2]
    reversed_groups=quality.group_candidates(list(reversed(rows)))
    assert list(reversed(reversed_groups))==groups


def test_all_empty_contexts_are_not_a_grouping_key():
    rows=[{"id":"a","prompt":"Cat","answer":"Animal"}, {"id":"b","prompt":"Oak","answer":"Tree"}]
    assert len(set(quality.group_candidates(rows)))==2


def test_null_optional_group_keys_do_not_merge_unrelated_rows():
    rows=[{"id":"a","prompt":"Cat","answer":"Animal","input_context":None,"source_group_id":None},
          {"id":"b","prompt":"Oak","answer":"Tree","input_context":None,"source_group_id":None}]
    assert len(set(quality.group_candidates(rows)))==2


@pytest.mark.parametrize('prompt',[
    '分解并简化表达式 8x³+12x²−4x。',
    'Find the unit fractions for 5/4.',
    'Explain a distribution with mean 50, median 55 and mode 60.',
    '求出多项式的因式。',
])
def test_out_of_scope_mathematical_variants_are_rejected(prompt):
    assert quality.quality_reason(prompt,'A complete but unverified answer.','en') in {'math_task','math_expression_task'}


def test_chinese_story_cut_in_middle_of_word_is_rejected():
    answer='小明每天认真学习,也帮助邻居解决困难。他走过许多地方,结识了许多朋友,并学会勇敢地面对生活。从那以后,亚'
    assert quality.quality_reason('写一个故事。',answer,'zh')=='possibly_truncated_prose'


def test_unsupported_summary_without_original_article_is_rejected():
    assert quality.quality_reason('总结这篇文章的主要内容。','文章中提到了几个重点。','zh')=='missing_source_material'
    assert quality.quality_reason('总结这篇文章的主要内容。\n小王每天步行上学,沿路观察植物,并记录天气。','小王在上学途中观察并记录自然现象。','zh') is None


def test_short_label_and_bullet_list_remain_allowed():
    assert quality.quality_reason('判断情感。','积极','zh') is None
    assert quality.quality_reason('列出用品。','- 纸张\n- 铅笔\n- 橡皮','zh') is None


def test_injury_advice_is_not_part_of_first_curriculum():
    assert quality.quality_reason('What grade is this muscle strain?','It is a severe injury.','en')=='specialist_medical_or_legal'


def test_formula_only_in_answer_does_not_escape_math_filter():
    assert quality.quality_reason('找出7个字母的可能组合数。','26 ^ 7 = 8,031,810,176','zh')=='math_task'
    assert quality.quality_reason('Explain the result.','The result is 1/4 plus 1/2.','en')=='math_expression_task'


@pytest.mark.parametrize('prompt', ['将此文本转换为大写字母','将此陈述归类为口头或非口头沟通的示例。'])
def test_missing_transformation_material_is_not_hallucinated(prompt):
    assert quality.quality_reason(prompt,'这是一个完整但没有根据的回答。','zh')=='missing_source_material'


def test_requesting_missing_material_and_inline_sentence_are_allowed():
    assert quality.quality_reason('从文章中提取几个关键点。','请您提供文章的内容。','zh') is None
    assert quality.quality_reason('猫跳上了沙发。用代词替换名词重写这个句子。','它跳上了沙发。','zh') is None


def test_incomplete_ambiguous_short_prompt_is_not_a_default_instruction():
    assert quality.quality_reason('约翰,读书。','阅读是一件很有意义的事情。','zh')=='underspecified_short_prompt'


def test_inline_programming_question_is_outside_first_curriculum():
    assert quality.quality_reason('Why does my C# program fail with dotnet run?','Check the project path.','en')=='code_task'


def test_group_invalid_limits():
    with pytest.raises(ValueError):
        quality.group_candidates([],threshold=0)
