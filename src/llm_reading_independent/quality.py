"""额外排除推荐广告、健康意见、过期办事日程以及可验证的日期矛盾。"""
from datetime import date
import re,unicodedata

QUESTION=re.compile(r'年货|天猫|淘宝|双十一|双11|暑假|寒假|放假|高考|中考|录取|招聘|工资|待遇|符文|出装|哪家|哪个品牌|什么牌子|最健康|最安全|养生|减肥|治疗|吃什么药|用什么药|医院哪|整形|丰胸|壮阳|(?:哪里|哪儿)[卖买]|什么.{0,12}好|哪个.{0,12}好')
ADVERTISEMENT=re.compile(r'微信公众号|公众平台|关注.*有礼|包吃包住|底薪|综合薪资|购买.*折|优惠券|在线咨询|免费咨询|点击咨询|咨询热线|加微信')
UNCERTAIN=re.compile(r'我不知道|名字我忘了|我没用过|我没用.+不知道')
WEEKDAY=re.compile(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*\((?:星期|周)([一二三四五六日天])\)')


def reason(row):
    question=row['question'];context=row['context']
    if row['language']=='zh':
        if QUESTION.search(question) or re.search(r'判多久|判几年|量刑|刑罚|判刑|坐牢',question):return 'recommendation_medical_or_ephemeral_query'
        if ADVERTISEMENT.search(context):return 'advertising_evidence'
        if UNCERTAIN.search(context):return 'explicitly_uncertain_evidence'
        if re.search(r'([\u4e00-\u9fff])\1{5}',context) or context.count('\\x0a')>3:return 'spam_or_badly_escaped_evidence'
        if re.search(r'临床实践证明|能使糖尿病|发病率下降|治愈率.*100%',context):return 'unsupported_clinical_promotion'
        for match in WEEKDAY.finditer(unicodedata.normalize('NFKC',context)):
            try:value=date(int(match[1]),int(match[2]),int(match[3]))
            except ValueError:return 'invalid_calendar_date'
            expected='一二三四五六日'[value.weekday()];given='日' if match[4]=='天' else match[4]
            if expected!=given:return 'contradictory_weekday'
    return None
