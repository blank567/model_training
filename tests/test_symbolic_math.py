import pytest
import sympy as sp
from llm_math_public.symbolic import parse_expression,verify,ParseError


def test_polynomial_precedence_fraction_and_implicit_product():
    x=sp.Symbol('x')
    assert sp.expand(parse_expression(r'-x^2+\frac{1}{2}x+2(x-1)')-(-x**2+x/2+2*x-2))==0


@pytest.mark.parametrize('bad',[
    "__import__('os').system('echo bad')",'x-','x^999999','1/0',r'\input{secret}',r'\sin{x}',
])
def test_unrecognized_or_unbounded_math_never_executes(bad):
    with pytest.raises((ParseError,ZeroDivisionError)):parse_expression(bad)


def test_recomputes_integral_from_question():
    prompt=r'计算定积分：$\int_{-5}^{1} (5 k - 2) \, dk$'
    good,reason=verify(prompt,r'原函数为二次多项式。最终答案 \boxed{-72}')
    assert reason is None and good['computed_answer']=='-72'
    assert verify(prompt,r'最终答案 \boxed{72}')[1]=='incorrect_final_answer'


def test_factor_and_expand_require_correct_operation_form():
    assert verify(r'因式分解：$x^2-1$',r'\boxed{(x-1)(x+1)}')[0]
    assert verify(r'因式分解：$x^2-1$',r'\boxed{x^2-1}')[1]=='answer_not_factorized'
    assert verify(r'展开：$(x-1)(x+1)$',r'\boxed{x^2-1}')[0]
    assert verify(r'展开：$(x-1)(x+1)$',r'\boxed{(x-1)(x+1)}')[1]=='answer_not_expanded'


def test_all_roots_required_not_just_last_box():
    prompt=r'解方程：$x^2-1=0$'
    assert verify(prompt,r'两个根 \boxed{-1} 和 \boxed{1}')[0]
    assert verify(prompt,r'\boxed{1}')[1]=='incorrect_or_incomplete_equation_roots'
    assert verify(r'解方程：$4a-13=5-8a$',r'\boxed{\frac{3}{2}}')[0]


def test_nested_power_and_variable_denominator_are_bounded():
    with pytest.raises(ParseError):parse_expression('((x+1)^8)^8')
    with pytest.raises(ParseError):parse_expression(r'\frac{x^2}{x}')


def test_wrong_intermediate_step_is_rejected_even_if_final_is_correct():
    result,reason=verify(r'展开：$(x+1)^2$',r'变形为 $2+2=5$，最后 \boxed{x^2+2x+1}')
    assert result is None and reason=='incorrect_intermediate_equality'


def test_variable_renaming_cannot_inflate_unique_problem_count():
    a,_=verify(r'展开：$(x+1)^2$',r'\boxed{x^2+2x+1}')
    b,_=verify(r'展开：$(s+1)^2$',r'\boxed{s^2+2s+1}')
    assert a['task_sha256']==b['task_sha256']
