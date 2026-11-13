"""有界多项式解析与独立验算。只构造白名单 SymPy 对象，不执行数据中的代码。

支持公开中文候选中的化简、展开、因式分解、一元方程与多项式定积分。
不把最终答案正确等同于自然语言推导全部正确；可解析的中间等式另作检查。
"""
from __future__ import annotations

import re
import unicodedata
import sympy as sp

from llm_lab.common.io import canonical_hash
from .rules import boxed_values


def math_spans(text):
    pattern=r'(?<!\\)\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$'
    return [next(v for v in match.groups() if v is not None).strip()
            for match in re.finditer(pattern,text,re.S)]


class ParseError(ValueError): pass


class PolynomialParser:
    def __init__(self,text):
        if len(text)>600:raise ParseError('Expression too long')
        text=text.replace('²','^{2}').replace('³','^{3}')
        text=unicodedata.normalize('NFKC',text).replace('−','-').replace('×','*').replace('÷','/')
        text=re.sub(r'\\(?:left|right|displaystyle)\b|\\[,;!]|\s+','',text)
        self.tokens=[];self.i=0;self.depth=0
        while text:
            match=re.match(r'\\(?:dfrac|tfrac|frac|sqrt)|\d+(?:\.\d+)?|[a-z]|[+*/^(){}-]',text)
            if match is None:raise ParseError('Unsupported mathematical syntax')
            self.tokens.append(match[0]);text=text[match.end():]
        if not self.tokens or len(self.tokens)>180:raise ParseError('Token limit')

    def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else None
    def take(self):
        value=self.peek()
        if value is None:raise ParseError('Unexpected end')
        self.i+=1;return value
    def parse(self):
        value=self.sum()
        if self.peek() is not None:raise ParseError('Unconsumed mathematical text')
        if len(value.free_symbols)>3 or sp.count_ops(value)>100:raise ParseError('Expression complexity limit')
        if value.has(sp.zoo,sp.oo,-sp.oo,sp.nan):raise ParseError('Undefined or unbounded expression')
        if degree_bound(value)>16:raise ParseError('Total symbolic degree limit')
        return value
    def sum(self):
        value=self.product()
        while self.peek() in {'+','-'}:
            op=self.take();other=self.product();value=value+other if op=='+' else value-other
        return value
    def product(self):
        value=self.unary()
        while True:
            token=self.peek()
            if token in {'*','/'}:
                op=self.take();other=self.unary()
                if op=='/' and other.free_symbols:raise ParseError('Variable denominator not supported')
                value=value*other if op=='*' else value/other
            elif token and (token in {'(','{',r'\frac',r'\dfrac',r'\tfrac',r'\sqrt'} or re.fullmatch(r'[a-z]|\d+(?:\.\d+)?',token)):
                value*=self.unary()
            else:break
        return value
    def unary(self):
        if self.peek() in {'+','-'}:
            op=self.take();value=self.unary();return value if op=='+' else -value
        value=self.atom()
        if self.peek()=='^':
            self.take();power=self.unary()
            if not power.is_Rational or abs(power)>8 or power.q>2:raise ParseError('Unsupported exponent')
            value=value**power
        return value
    def atom(self):
        token=self.take()
        if token in {'(','{'}:
            self.depth+=1
            if self.depth>20:raise ParseError('Nesting limit')
            value=self.sum();close=')' if token=='(' else '}'
            if self.take()!=close:raise ParseError('Unbalanced grouping')
            self.depth-=1;return value
        if token in {r'\frac',r'\dfrac',r'\tfrac'}:
            numerator=self.atom();denominator=self.atom()
            if denominator.free_symbols:raise ParseError('Variable denominator not supported')
            return numerator/denominator
        if token==r'\sqrt':return sp.sqrt(self.atom())
        if re.fullmatch('[a-z]',token):return sp.Symbol(token)
        if re.fullmatch(r'\d+(?:\.\d+)?',token):
            value=sp.Rational(token)
            if value>10**8:raise ParseError('Numeric magnitude limit')
            return value
        raise ParseError('Unexpected token')


def parse_expression(text):return PolynomialParser(text.strip().strip('$')).parse()


def degree_bound(expr):
    if expr.is_Number:return 0
    if expr.is_Symbol:return 1
    if expr.is_Add:return max(degree_bound(value) for value in expr.args)
    if expr.is_Mul:return sum(degree_bound(value) for value in expr.args)
    if expr.is_Pow:
        base,power=expr.args
        if not base.free_symbols:return 0
        if not power.is_Integer or power<0:raise ParseError('Non-polynomial power')
        return int(power)*degree_bound(base)
    raise ParseError('Unsupported symbolic object')


def bounded_poly(expr):
    symbols=sorted(expr.free_symbols,key=str)
    if not symbols or len(symbols)>2:raise ParseError('One or two polynomial variables required')
    try:poly=sp.Poly(expr,*symbols,domain=sp.QQ)
    except (sp.PolynomialError,sp.CoercionFailed) as error:raise ParseError('Rational-coefficient polynomial required') from error
    if poly.total_degree()>6 or len(poly.terms())>35:raise ParseError('Polynomial degree/term limit')
    return poly,symbols


def roots(expr,var):
    poly=sp.Poly(expr,var,domain=sp.QQ)
    if poly.degree()>3:raise ParseError('Equation degree limit')
    if poly.degree()<=2:return set(sp.solve(poly.as_expr(),var))
    rational=sp.polys.polytools.ground_roots(poly)
    if sum(rational.values())!=poly.degree():raise ParseError('Cubic has non-rational roots')
    return set(rational)


def problem(prompt):
    spans=math_spans(prompt)
    if len(spans)!=1:raise ParseError('A single self-contained mathematical expression is required')
    source=spans[0]
    outside=prompt.replace(source,'')
    if re.search(r'分别|下列|证明|判断|是否|几何|图像|选项|已知|个数|取值|求值|代入',outside):raise ParseError('Unsupported compound question')
    if '定积分' in prompt:
        text=re.sub(r'\\(?:left|right|displaystyle)\b','',source).strip()
        match=re.fullmatch(r'\\int_\{([^{}]+)\}\^\{([^{}]+)\}\s*(.*?)\s*(?:\\[,;!])?\s*d\s*([a-z])',text,re.S)
        if match is None:raise ParseError('Unsupported definite integral layout')
        low,high,integrand,var=match.groups();var=sp.Symbol(var)
        low,high=parse_expression(low),parse_expression(high)
        expr=parse_expression(integrand);poly,symbols=bounded_poly(expr)
        if symbols!=[var] or low.free_symbols or high.free_symbols:raise ParseError('Univariate constant-bound integral required')
        primitive=poly.integrate().as_expr();expected=sp.expand(primitive.subs(var,high)-primitive.subs(var,low))
        task='definite_integral'
        return {'task':task,'expression':expr,'symbols':symbols,'expected':expected,'primitive':primitive,
                'bounds':[low,high],'topic':'calculus'}
    if '=' in source:
        if not re.search(r'方程|求解|解出',prompt) or source.count('=')!=1:raise ParseError('Unsupported equation task')
        left,right=source.split('=');expr=parse_expression(left)-parse_expression(right);poly,symbols=bounded_poly(expr)
        if len(symbols)!=1:raise ParseError('Equation must have one variable')
        return {'task':'solve_equation','expression':expr,'symbols':symbols,'expected':roots(expr,symbols[0]),'topic':'algebra_equations'}
    expr=parse_expression(source);poly,symbols=bounded_poly(expr)
    if re.search('因式分解|因子分解',prompt):task='factor';expected=sp.factor(expr)
    elif '展开' in prompt:task='expand';expected=sp.expand(expr)
    elif re.search('化简|简化|合并同类项',prompt):task='simplify';expected=sp.expand(expr)
    else:raise ParseError('Unsupported operation')
    return {'task':task,'expression':expr,'symbols':symbols,'expected':expected,'topic':'algebra_equations'}


def candidate_values(text):
    text=text.strip().strip('$').strip()
    text=re.sub(r'\\(?:,|;|quad|qquad)',' ',text)
    text=re.sub(r'^\\[\{]','',text);text=re.sub(r'\\[\}]$','',text)
    if text.startswith('{') and text.endswith('}'):text=text[1:-1]
    text=text.replace('，',',').replace('、',',')
    parts=re.split(r',|或|\\text\{或\}',text)
    result=[]
    for part in parts:
        part=part.strip()
        if '=' in part:
            lhs,part=part.rsplit('=',1)
            if re.fullmatch(r'[a-z](?:_\{?\d+\}?)?',lhs.strip()) is None:raise ParseError('Unsupported root assignment')
        result.append(parse_expression(part))
    return result


def verify(prompt,response):
    """成功返回计算依据；失败保留原因，不能将未知解析当成验证通过。"""
    try:
        spec=problem(prompt);boxes=boxed_values(response)
        if not boxes:raise ParseError('Explicit boxed final answer required')
        if spec['task']=='solve_equation':
            candidates=[]
            # 多个boxed根必须共同保留；只抓最后一个会丢失多答案。
            for value in boxes:candidates.extend(candidate_values(value))
            actual=set(candidates)
            if actual!=spec['expected']:return None,'incorrect_or_incomplete_equation_roots'
            answer=', '.join(sp.sstr(x) for x in sorted(actual,key=sp.default_sort_key))
        else:
            value=boxes[-1]
            if '=' in value:
                pieces=value.split('=')
                values=[parse_expression(piece) for piece in pieces]
                if any(sp.expand(a-b)!=0 for a,b in zip(values,values[1:])):return None,'false_final_equality'
                actual=values[-1]
            else:actual=parse_expression(value)
            if sp.expand(actual-spec['expected'])!=0:return None,'incorrect_final_answer'
            if spec['task']=='expand' and actual!=sp.expand(actual):return None,'answer_not_expanded'
            if spec['task']=='factor' and spec['expected'].is_Mul and actual.is_Add:return None,'answer_not_factorized'
            answer=sp.sstr(spec['expected'])
        # 核对能完整解析的展示等式；方程变换按解集核对，恒等变形按多项式核对。
        checked=0
        for span in math_spans(response):
            if '=' not in span or '\\boxed' in span or '\\begin' in span:continue
            parts=span.split('=')
            for left,right in zip(parts,parts[1:]):
                try:
                    a,b=parse_expression(left),parse_expression(right)
                    if not (a.free_symbols|b.free_symbols).issubset(set(spec['symbols'])):continue
                    if spec['task']!='solve_equation' and a.is_Symbol and not b.free_symbols:continue
                    if spec['task']=='solve_equation' and a.free_symbols|b.free_symbols:
                        if roots(a-b,spec['symbols'][0])!=spec['expected']:return None,'incorrect_intermediate_equation'
                    elif sp.expand(a-b)!=0:return None,'incorrect_intermediate_equality'
                    checked+=1
                except (ParseError,sp.PolynomialError,sp.CoercionFailed):continue
        rename={symbol:sp.Symbol('v'+str(i)) for i,symbol in enumerate(spec['symbols'])}
        canonical={'task':spec['task'],'polynomial':sp.srepr(sp.expand(spec['expression'].xreplace(rename))),
                   'bounds':[sp.sstr(x) for x in spec.get('bounds',[])]}
        return {'verifier':'bounded_polynomial_sympy_1.14_v1','task':spec['task'],'topic':spec['topic'],
                'canonical_task':canonical,'task_sha256':canonical_hash(canonical),'computed_answer':answer,
                'intermediate_equalities_checked':checked,'final_answer_verified':True,
                'full_natural_language_proof_certified':False},None
    except (ParseError,ValueError,TypeError,ZeroDivisionError,sp.PolynomialError,sp.CoercionFailed) as error:
        return None,'unsupported_or_invalid:'+str(error)[:100]
