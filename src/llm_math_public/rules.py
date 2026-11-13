"""可审查的质量规则；学科标签是启发式，不能冒充人工题型或难度标注。"""
import re
import unicodedata
import hashlib

TOPIC_PATTERNS = [
    ('calculus', r'\b(?:derivative|differentiat\w*|integral|integrat\w*|differential equation)\b|\\(?:lim|int)\b|d/dx|极限|导数|求导|积分|微分'),
    ('linear_algebra', r'\b(?:matrix|matrices|determinant|eigen\w*|vector space|linear transformation)\b|矩阵|行列式|特征值|特征向量|向量空间'),
    ('probability_statistics', r'\b(?:probability|random\w*|variance|standard deviation|median|distribution|statistics|expected value)\b|概率|随机|方差|标准差|中位数|统计|期望'),
    ('combinatorics', r'\b(?:permut\w*|combinatori\w*|combination\w*|arrangement\w*|how many ways|number of ways|distinct ways)\b|排列|组合|种方式|种方法'),
    ('number_theory', r'\b(?:prime\w*|divisib\w*|congru\w*|modulo|remainder|greatest common divisor|least common multiple|gcd|lcm)\b|质数|素数|整除|同余|余数|最大公约数|最小公倍数'),
    ('geometry', r'\b(?:triang\w*|circle\w*|angle\w*|rectang\w*|polygon\w*|sphere\w*|cylinder\w*|diameter|radius|radii|quadrilateral\w*|perimeter|geometry|area|volume)\b|三角形|圆|角度|矩形|多边形|球体|圆柱|直径|半径|四边形|周长|几何|面积|体积'),
    ('functions_trigonometry', r'\b(?:function\w*|trigonom\w*|sine|cosine|tangent|logarith\w*|sequence|series|progression)\b|\\(?:sin|cos|tan|log|ln)\b|\b[fg]\s*\(\s*x|函数|数列|三角函数|对数'),
    ('algebra_equations', r'\b(?:algebra\w*|equation\w*|inequal\w*|polynomial\w*|quadratic\w*|simplif\w*|expression\w*|solve for)\b|方程|不等式|多项式|二次|化简|代数|[xyz]\s*(?:\^|\*\*|[+\-=])'),
]
TOPICS = [item[0] for item in TOPIC_PATTERNS] + ['arithmetic_word']
DIAGRAM = re.compile(r'(?:shown|following|given).{0,25}(?:figure|diagram|picture)|in the figure|diagram (?:below|above)|as illustrated|\\includegraphics|<image>|<img|如图|图中|下图|见图', re.I)
CODE = re.compile(r'```(?:python|py)|(?m:^\s*(?:import |from \w+ import |def \w+\(|print\())')
PHYSICS = re.compile(r'\b(?:newtons?|joules?|voltage|electric field|magnetic|thermodynamic\w*|quantum|kinetic energy|molar|electrons?|protons?)\b', re.I)


def classify(prompt, source=''):
    # integral powers 表示整数次幂，不是微积分中的积分。
    prompt=re.sub(r'\bintegral (?=(?:powers?|values?|solutions?|coordinates?|roots?|points?)\b)','integer ',prompt,flags=re.I)
    for topic, pattern in TOPIC_PATTERNS:
        if re.search(pattern, prompt, re.I):
            return topic
    if re.search(r'gsm8k|mathqa|aqua_rat', source, re.I):
        return 'arithmetic_word'
    return None


def reject_reason(prompt, response, source, language):
    if not isinstance(prompt, str) or not isinstance(response, str) or not prompt.strip() or not response.strip():
        return 'empty_or_invalid_text'
    if '/pot/' in source.lower() or CODE.search(response):
        return 'program_of_thought_not_executed'
    if DIAGRAM.search(prompt):
        return 'requires_missing_diagram'
    if PHYSICS.search(prompt):
        return 'non_math_physics_or_chemistry'
    if '\ufffd' in prompt + response or len(response.strip()) < 15:
        return 'corrupt_or_insufficient_solution'
    if re.search(r'as an ai|cannot solve|unable to solve|无法解答|无法回答|无法确定答案|不能解答', response, re.I):
        return 'refusal_or_missing_solution'
    if language == 'zh' and len(re.findall('[\u4e00-\u9fff]', prompt)) < 3:
        return 'not_chinese_prompt'
    return None


def boxed_values(text):
    """平衡花括号，保留分式等嵌套 LaTeX，拒绝截断的 boxed。"""
    result = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            if text[i] == '}': depth -= 1
            if depth == 0:
                result.append(text[start:i].strip())
                break
    return result


def final_answer(text):
    values = boxed_values(text)
    if values:
        return values[-1]
    matches = list(re.finditer(r'(?:the (?:final )?answer is|(?:最终)?答案\s*(?:是|为|[:：]))\s*([^\n]+)', text, re.I))
    if not matches:
        return None
    value = matches[-1].group(1).strip().strip('$').strip()
    value = re.sub(r'[。.]\s*$', '', value).strip()
    if not value or len(value) > 160:
        return None
    return value


def answer_key(value):
    if value is None:
        return None
    value = unicodedata.normalize('NFKC', value).strip().strip('$').strip()
    value = re.sub(r'\\(?:left|right|,|!|;|quad|qquad)|\s+', '', value)
    value = value.replace('\\dfrac', '\\frac').replace('−', '-')
    choice = re.fullmatch(r'\(?([A-H])\)?(?:[.。])?', value)
    return choice.group(1) if choice else value


def numeric_literals(text):
    return set(re.findall(r'\d+(?:\.\d+)?', unicodedata.normalize('NFKC', text)))


def question_hash(text,converter=None):
    if converter is not None:text=converter.convert(text)
    text=re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',text).casefold())
    return hashlib.sha256(text.encode()).hexdigest()
