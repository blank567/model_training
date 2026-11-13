"""八类一至两步应用题；中英文是同一数学题的两个表述，始终一起划分。"""
from __future__ import annotations

from .verifier import expression_key


FAMILIES = ('add', 'subtract', 'multiply', 'divide', 'multiply_add', 'multiply_subtract',
            'add_multiply', 'subtract_multiply')

# 0/1用于训练和ID测试，2仅用于开发，3/4保留给模板泛化测试。
TEXT = {
 'add': {
  'zh': ('盒子里原有{a}支铅笔，又放入{b}支，现在有多少支？',
         '上午卖出{a}本书，下午卖出{b}本书，一天共卖出多少本？',
         '小林收集了{a}张卡片，朋友送来{b}张，合计多少张？',
         '仓库第一批收到{a}件货物，第二批收到{b}件，两批一共多少件？',
         '甲组植树{a}棵，乙组植树{b}棵，两组共植树多少棵？'),
  'en': ('A box has {a} pencils. Another {b} pencils are added. How many pencils are there now?',
         'A shop sold {a} books in the morning and {b} in the afternoon. How many books did it sell in total?',
         'Lin collected {a} cards and received {b} more from a friend. How many cards does Lin have?',
         'A warehouse received {a} items in its first shipment and {b} in its second. How many items arrived?',
         'Team A planted {a} trees and team B planted {b} trees. How many trees did they plant altogether?')},
 'subtract': {
  'zh': ('盒子里有{a}支铅笔，取走{b}支，还剩多少支？',
         '商店原有{a}本书，卖出{b}本后还剩多少本？',
         '小林有{a}张卡片，送出{b}张后还剩多少张？',
         '一段路共{a}米，已经走了{b}米，还要走多少米？',
         '仓库有{a}件货物，发走{b}件，库存还剩多少件？'),
  'en': ('A box holds {a} pencils. After {b} are removed, how many remain?',
         'A shop had {a} books and sold {b}. How many books are left?',
         'Lin has {a} cards and gives away {b}. How many cards remain?',
         'A path is {a} meters long. After walking {b} meters, how many meters are left?',
         'A warehouse had {a} items and shipped {b}. How many items remain in stock?')},
 'multiply': {
  'zh': ('每盒有{a}支铅笔，{b}盒共有多少支？',
         '一本书{a}元，买{b}本共需要多少元？',
         '每排有{a}个座位，{b}排共有多少个座位？',
         '机器每小时加工{a}件产品，工作{b}小时共加工多少件？',
         '每袋装{a}颗糖，{b}袋一共装多少颗糖？'),
  'en': ('Each box has {a} pencils. How many pencils are in {b} boxes?',
         'A book costs {a} dollars. What is the total cost of {b} books?',
         'There are {a} seats per row. How many seats are in {b} rows?',
         'A machine makes {a} items per hour. How many items does it make in {b} hours?',
         'Each bag contains {a} sweets. How many sweets are in {b} bags?')},
 'divide': {
  'zh': ('把{a}支铅笔平均放进{b}个盒子，每盒放多少支？',
         '把{a}本书平均分给{b}个班，每班分到多少本？',
         '{b}个小组平均分{a}张卡片，每组分到多少张？',
         '工人用{b}小时均匀加工{a}件产品，每小时加工多少件？',
         '把{a}颗糖平均装进{b}袋，每袋有多少颗？'),
  'en': ('There are {a} pencils shared equally among {b} boxes. How many pencils go in each box?',
         'There are {a} books shared equally among {b} classes. How many books does each class get?',
         'There are {a} cards shared equally by {b} teams. How many cards does each team receive?',
         'A worker makes {a} items at a steady rate over {b} hours. How many items are made per hour?',
         'There are {a} sweets divided equally into {b} bags. How many sweets are in each bag?')},
 'multiply_add': {
  'zh': ('每盒有{a}支铅笔，买了{b}盒，另买{c}支，共有多少支？',
         '每套书有{a}本，收到{b}套，另外收到{c}本，共收到多少本？',
         '每组有{a}人，{b}个组之外另有{c}人，一共有多少人？',
         '机器每小时加工{a}件，工作{b}小时后另加工{c}件，共加工多少件？',
         '{b}袋糖每袋有{a}颗，另外有{c}颗散装糖，一共多少颗？'),
  'en': ('Each box has {a} pencils. You buy {b} boxes and {c} extra pencils. How many pencils do you buy?',
         'Each book set has {a} books. A library receives {b} sets and {c} extra books. How many books arrive?',
         'There are {b} teams of {a} people each and {c} other people. How many people are there in total?',
         'A machine makes {a} items per hour for {b} hours, then makes {c} extra items. How many items does it make?',
         'There are {b} bags of {a} sweets each and {c} loose sweets. How many sweets are there?')},
 'multiply_subtract': {
  'zh': ('每盒有{a}支铅笔，原有{b}盒，取走{c}支后还剩多少支？',
         '每套书有{a}本，原有{b}套，借出{c}本后还剩多少本？',
         '有{b}排座位，每排{a}个，其中{c}个被占用，还空着多少个？',
         '机器每小时加工{a}件，工作{b}小时，其中{c}件不合格，合格产品有多少件？',
         '{b}袋糖每袋{a}颗，送出{c}颗后还剩多少颗？'),
  'en': ('There are {b} boxes with {a} pencils each. After {c} pencils are removed, how many remain?',
         'There are {b} book sets with {a} books each. After {c} books are borrowed, how many remain?',
         'There are {b} rows of {a} seats each. If {c} seats are occupied, how many are empty?',
         'A machine makes {a} items per hour for {b} hours. If {c} items are defective, how many are good?',
         'There are {b} bags with {a} sweets each. After giving away {c} sweets, how many remain?')},
 'add_multiply': {
  'zh': ('每个盒子放{a}支红铅笔和{b}支蓝铅笔，{c}个盒子共有多少支铅笔？',
         '每套书有{a}本故事书和{b}本科普书，{c}套共有多少本书？',
         '每组有{a}名男生和{b}名女生，{c}组一共有多少名学生？',
         '一份套餐有{a}个苹果和{b}个梨，{c}份套餐一共有多少个水果？',
         '每袋装{a}颗奶糖和{b}颗水果糖，{c}袋共有多少颗糖？'),
  'en': ('Each box has {a} red pencils and {b} blue pencils. How many pencils are in {c} boxes?',
         'Each set has {a} storybooks and {b} science books. How many books are in {c} sets?',
         'Each group has {a} boys and {b} girls. How many students are in {c} groups?',
         'Each pack has {a} apples and {b} pears. How many pieces of fruit are in {c} packs?',
         'Each bag has {a} milk sweets and {b} fruit sweets. How many sweets are in {c} bags?')},
 'subtract_multiply': {
  'zh': ('每盒原有{a}支铅笔，每盒取走{b}支，{c}盒共剩多少支？',
         '每套书原有{a}本，每套借出{b}本，{c}套共剩多少本？',
         '每排有{a}个座位，每排有{b}个被占用，{c}排共有多少个空座位？',
         '每天加工{a}件产品，其中每天{b}件不合格，{c}天共得到多少件合格产品？',
         '每袋原有{a}颗糖，每袋取走{b}颗，{c}袋共剩多少颗？'),
  'en': ('Each box originally has {a} pencils. After removing {b} pencils per box, how many remain in {c} boxes?',
         'Each set originally has {a} books. After {b} books per set are borrowed, how many remain in {c} sets?',
         'Each row has {a} seats and {b} occupied seats. How many empty seats are there in {c} rows?',
         'A factory makes {a} items daily, of which {b} are defective each day. How many good items are made in {c} days?',
         'Each bag originally has {a} sweets. After removing {b} sweets per bag, how many remain in {c} bags?')},
}


def make_problem(family, rng, template_index):
    a, b, c = rng.randint(2, 199), rng.randint(2, 199), rng.randint(2, 40)
    if family == 'add': expression, answer = f'{a}+{b}', a + b
    elif family == 'subtract':
        a = rng.randint(3, 499)
        b = rng.randint(1, a - 1)
        expression, answer = f'{a}-{b}', a - b
    elif family == 'multiply':
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        expression, answer = f'{a}*{b}', a * b
    elif family == 'divide':
        quotient, b = rng.randint(2, 199), rng.randint(2, 99)
        a = quotient * b
        expression, answer = f'{a}/{b}', quotient
    elif family == 'multiply_add': expression, answer = f'{a}*{b}+{c}', a * b + c
    elif family == 'multiply_subtract':
        c = rng.randint(1, min(a * b - 1, 199))
        expression, answer = f'{a}*{b}-{c}', a * b - c
    elif family == 'add_multiply': expression, answer = f'({a}+{b})*{c}', (a + b) * c
    elif family == 'subtract_multiply':
        a = rng.randint(3, 199)
        b = rng.randint(1, a - 1)
        expression, answer = f'({a}-{b})*{c}', (a - b) * c
    else: raise ValueError('Unknown math family')
    prompts = {lang: TEXT[family][lang][template_index].format(a=a, b=b, c=c) for lang in ('zh', 'en')}
    common = [answer - 1, answer + 1, a + b, abs(a - b), a * b, a, b, answer - b, answer + b]
    negatives = sorted({value for value in common if value >= 0 and value != answer and len(str(value)) == len(str(answer))})
    if not negatives:
        negatives = [answer + 1]
    wrong = rng.choice(negatives)
    return {'family': family, 'difficulty': 'one_step' if family in FAMILIES[:4] else 'two_step',
            'template_id': template_index, 'expression': expression, 'ground_truth': str(answer),
            'rejected_answer': str(wrong), 'expression_key': expression_key(expression),
            'prompts': prompts, 'max_operand': max((a, b) if family in FAMILIES[:4] else (a, b, c))}
