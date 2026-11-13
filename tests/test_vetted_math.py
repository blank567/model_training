from copy import deepcopy
from types import SimpleNamespace
from llm_math_public.vetted import parse_row,formula_key

CONVERTER=SimpleNamespace(convert=lambda text:text)


def numina():
    return {'problem':'Find the derivative of f(x)=x^2 at x=3.',
        'solution':'Differentiation gives f prime equal to 2x. The answer is 6.',
        'answer':'6','problem_type':'Calculus','question_type':'math-word-problem',
        'problem_is_valid':'Yes','solution_is_valid':'Yes','synthetic':False,'source':'cn_contest'}


def test_publisher_validity_and_non_synthetic_are_required():
    for key,value in [('solution_is_valid','No'),('problem_is_valid','Unknown'),('synthetic',True)]:
        row=numina();row[key]=value
        assert parse_row(row,'numina',CONVERTER)[0] is None


def test_correct_final_answer_is_bound_to_reference_solution():
    good,reason=parse_row(numina(),'numina',CONVERTER)
    assert reason is None and good['final_answer']=='6' and good['response'].endswith('Answer: 6')
    bad=numina();bad['answer']='7'
    assert parse_row(bad,'numina',CONVERTER)==(None,'reference_numeric_disagrees_with_gold')


def test_proof_marker_is_not_trained_as_literal_final_answer():
    row=numina();row['answer']='proof';row['question_type']='proof'
    got,reason=parse_row(row,'numina',CONVERTER)
    assert reason is None and got['final_answer'] is None and not got['response'].endswith('proof')


def test_cmm_question_with_image_is_never_silently_made_text_only():
    row={'question':'这道题目的图像提供了主要几何关系，请选择正确答案。','analysis':'完整的几何推导过程，最后得到答案为二。',
         'answer':'2','image':'["figure.png"]','options':'','subject':'几何','level':'高一'}
    assert parse_row(row,'cmm',CONVERTER)==(None,'requires_image')


def test_formula_dedup_requires_substantial_content():
    assert formula_key('A value is $x$.') is None
    a=r'Solve $x^{2}+2x+1=0$ and $y^{2}+4y+4=0$ for the pair.'
    b=r'求解 $x^{2}+2x+1=0$ 及 $y^{2}+4y+4=0$ 的解。'
    assert formula_key(a)==formula_key(b)


def test_multi_eigenvalue_answer_cannot_be_only_last_box():
    row=numina();row['problem']='Find all eigenvalues of the matrix.'
    row['solution']=r'The eigenvalues computed from the determinant are \boxed{2} and \boxed{3}.'
    row['answer']='3'
    assert parse_row(row,'numina',CONVERTER)==(None,'incomplete_multi_answer_gold')


def test_undefined_numbered_theorem_context_is_not_self_contained():
    row=numina();row['solution']='By Theorem 3 we get the expression immediately. The answer is 6.'
    assert parse_row(row,'numina',CONVERTER)==(None,'missing_numbered_theorem_context')
