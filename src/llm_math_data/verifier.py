"""严格数值奖励与独立表达式验算；不从一串候选数字中猜测最终答案。"""
from __future__ import annotations

import ast
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import re
import unicodedata


NUMBER = re.compile(r'[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:/[+-]?[0-9]+)?\Z')


def numeric_value(text):
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize('NFKC', text).strip()
    if len(text) > 48 or not NUMBER.fullmatch(text):
        return None
    try:
        if '/' in text:
            numerator, denominator = text.replace(',', '').split('/')
            return Fraction(Decimal(numerator)) / int(denominator)
        return Fraction(Decimal(text.replace(',', '')))
    except (ValueError, ArithmeticError):
        return None


def reward(completion, ground_truth, reward_type='numeric_exact'):
    """只有真实答案正确才得1分，不额外奖励标签、长度或思维链格式。"""
    if reward_type == 'numeric_exact':
        expected = numeric_value(ground_truth)
        if expected is None:
            raise ValueError('Invalid numeric reference')
        return float(numeric_value(completion) == expected)
    if reward_type == 'choice_exact':
        expected = str(ground_truth).strip().upper()
        if not re.fullmatch('[A-H]', expected):
            raise ValueError('Invalid single-choice reference')
        actual = unicodedata.normalize('NFKC', completion).strip().upper()
        return float(actual == expected)
    raise ValueError('Unknown reward type')


def accuracy_reward(completions, ground_truth, reward_type=None, **kwargs):
    """TRL形式的适配函数；ground_truth/规则是奖励侧字段，不进入模型prompt。"""
    if len(completions) != len(ground_truth):
        raise ValueError('Completion/reference counts differ')
    kinds = reward_type or ['numeric_exact'] * len(completions)
    if len(kinds) != len(completions):
        raise ValueError('Reward type count differs')
    result = []
    for completion, answer, kind in zip(completions, ground_truth, kinds):
        if isinstance(completion, list):
            completion = completion[-1]['content'] if len(completion) == 1 else ''
        result.append(reward(completion, answer, kind))
    return result


def expression_value(expression):
    """用白名单AST和有理数重算标签；禁止eval、函数调用、幂和外部名称。"""
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 50:
        raise ValueError('Expression too complex')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 1000000:
            return Fraction(node.value)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('Unsupported expression')
    return visit(tree)


def expression_key(expression):
    # 交换加法/乘法的操作数不会让同一算式变成另一个独立题目。
    def canonical(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return str(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            pair = [canonical(node.left), canonical(node.right)]
            if isinstance(node.op, (ast.Add, ast.Mult)):
                pair.sort(key=lambda value: json.dumps(value, sort_keys=True))
            return [type(node.op).__name__, pair]
        raise ValueError('Unsupported expression key')
    expression_value(expression)
    return hashlib.sha256(json.dumps(canonical(ast.parse(expression, mode='eval').body), sort_keys=True).encode()).hexdigest()


def question_hash(text, converter):
    text = converter.convert(text)
    text = re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text).casefold())
    return hashlib.sha256(text.encode()).hexdigest()
