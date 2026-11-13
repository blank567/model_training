from types import SimpleNamespace
import pytest

from llm_math_public.rules import classify,reject_reason,boxed_values,answer_key,question_hash
from llm_math_public.build import clean_text,allocate
from llm_math_data.verifier import question_hash as legacy_question_hash


@pytest.mark.parametrize('prompt,expected',[
    ('Find the derivative of sin(x).','calculus'),
    ('Calculate the determinant of this matrix.','linear_algebra'),
    ('Find the probability of two independent events.','probability_statistics'),
    ('In how many ways can the letters be arranged?','combinatorics'),
    ('Find the remainder modulo 7.','number_theory'),
    ('What is the area of a triangle?','geometry'),
    ('Find the domain of the function f(x).','functions_trigonometry'),
    ('Solve for x in the quadratic equation.','algebra_equations'),
])
def test_core_subject_coverage(prompt,expected):
    assert classify(prompt)==expected


def test_unknown_is_not_silently_relabelled_as_math():
    assert classify('Tell a story about a kitten.') is None
    assert classify('Find the terms with positive integral powers of x.')!='calculus'


def test_literal_newline_cleanup_preserves_latex_commands():
    assert clean_text(r'line\n$\nu+\nabla f$')=='line\n$\\nu+\\nabla f$'
    assert clean_text(r'\nabla')==r'\nabla'


def test_boxed_fraction_balanced_braces_not_last_digit_matching():
    assert boxed_values(r'a=\boxed{\frac{2}{3}}, b=\boxed{4}')==[r'\frac{2}{3}','4']
    assert boxed_values(r'\boxed{\frac{2}{3}')==[]
    assert answer_key(r'\dfrac{2}{3}')==answer_key(r'\frac{2}{3}')


def test_missing_diagram_and_code_rejected():
    assert reject_reason('As shown in the figure, find x.','Here is a complete derivation.','CoT','en')=='requires_missing_diagram'
    assert reject_reason('Find x.','import os\nprint(1)','CoT','en')=='program_of_thought_not_executed'
    assert reject_reason('Find x.','A normal answer with enough text.','data/PoT/math.json','en')=='program_of_thought_not_executed'


def test_decontamination_hash_remains_compatible_with_previous_release():
    converter=SimpleNamespace(convert=lambda text:text)
    assert question_hash('Find x: 1 + 2?')==legacy_question_hash('Find x: 1 + 2?',converter)
    assert question_hash('Find x: 1 + 2?')==question_hash('FIND X 1+2')


def test_rare_subject_capacity_is_respected():
    q=allocate(20,{'a':3,'b':20},{'a':.5,'b':.5},minimum=1)
    assert q=={'a':3,'b':17}
    with pytest.raises(ValueError):allocate(25,{'a':3,'b':20},{'a':.5,'b':.5})


def test_quarantined_dataset_cannot_be_republished(tmp_path):
    from llm_math_public.audit import audit
    (tmp_path/'DO_NOT_TRAIN.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='Quarantined'):
        audit(tmp_path/'manifest.json',tmp_path,publish=True)
