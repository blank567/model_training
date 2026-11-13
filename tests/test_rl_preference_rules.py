from llm_rl.preference_rules import acceptable_negative


def source(**changes):
    row={'language':'en','reference_answers':['John'],'context':'John Doubleday lived in London, England.'};row.update(changes);return row


def test_policy_negative_is_a_different_material_span():
    assert acceptable_negative([10,2],'London',source())
    assert not acceptable_negative([10,2],'John',source())
    assert not acceptable_negative([10,2],'John Doubleday',source())
    assert acceptable_negative([10,2],'Paris',source())  # 任务明确要求材料内片段：这是格式负例。
    assert not acceptable_negative([10,2],'Paris',source(),require_grounded=True)
    assert not acceptable_negative([10],'London',source())


def test_punctuation_diacritics_and_partial_aliases_are_not_negative():
    assert not acceptable_negative([10,2],'Francois',source(reference_answers=['François'],context='François and Francois visited London.'))
    assert not acceptable_negative([10,2],'United States',source(reference_answers=['the United States'],context='United States'))


def test_simplified_chinese_negative_is_grounded():
    row=source(language='zh',reference_answers=['北京大学'],context='北京大学位于北京，复旦大学位于上海。')
    assert acceptable_negative([10,2],'上海',row)
    assert not acceptable_negative([10,2],'北京',row)
    assert not acceptable_negative([10,2],'清华大学',row,require_grounded=True)
