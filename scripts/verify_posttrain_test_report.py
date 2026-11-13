"""核验交付报告的全量数据、统计、文件SHA及离线HTML链接完整性。"""
from pathlib import Path
from html.parser import HTMLParser
import argparse,base64,csv,hashlib,json,math


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10),(a,b)


class ReportHTML(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.images=[]
    def handle_starttag(self,tag,attributes):
        attrs=dict(attributes)
        if 'id' in attrs:self.ids.append(attrs['id'])
        if tag=='a':self.links.append(attrs.get('href',''))
        if tag=='img':self.images.append(attrs.get('src',''))


def verify(work):
    work=Path(work);data=work/'results_local';report=work/'report';plan=read(work/'sources_local/plan.json');comparison=read(data/'comparison.json')
    assert comparison['status']=='completed' and comparison['plan_identity']==plan['plan_identity']
    generated=0;loss_rows=0;test_ids={};scores={}
    for suite,results in comparison['results'].items():
        spec=plan['suites'][suite];expected=spec['expected_examples'];metadata={r['id']:r for r in lines(data/suite/'test_metadata.jsonl')}
        assert len(metadata)==expected
        for result in results:
            directory=data/suite/result['key'];rows=lines(directory/'generations.jsonl');losses=lines(directory/'test_rows.jsonl')
            assert len(rows)==len(losses)==expected
            assert {r['id'] for r in rows}=={r['id'] for r in losses}==set(metadata)
            assert len({r['id'] for r in rows})==expected
            for r in rows:
                ids=r['generated_ids'];assert ids and len(ids)<=spec['max_new_tokens']
                assert r['stop_reason'] in ['eos','max_new_tokens']
                assert (r['stop_reason']=='eos')==(ids[-1]==2)
                if r['stop_reason']=='max_new_tokens':assert len(ids)==spec['max_new_tokens']
                assert 2 not in ids[:-1]
                assert not any(token in spec['forbidden_tokens'] for token in ids)
                assert metadata[r['id']]['prompt_token_count']+len(ids)<=1024
            macro=[]
            for language in ['zh','en']:
                values=[r for r in losses if r['language']==language]
                tokens=sum(r['tokens'] for r in values);loss=sum(r['loss_sum'] for r in values)/tokens
                close(loss,result['test'][language]['loss']);macro.append(loss)
            close(sum(macro)/2,result['test']['macro_loss'])
            if suite=='reading_test_2000':
                em=sum(r['exact_match'] for r in rows)/len(rows);close(em,result['generation']['macro_exact_match']);scores[result['key']]=em
                close(sum(r['stop_reason']=='eos' for r in rows)/len(rows),result['generation']['eos_rate'])
            else:close(sum(r['stop_reason']=='eos' for r in rows)/len(rows),result['generation']['eos_stop_fraction'])
            generated+=len(rows);loss_rows+=len(losses)
        test_ids[suite]=set(metadata)
    assert generated==59000 and len(set.union(*test_ids.values()))==3000
    cross=read(data/'cross_domain_loss/comparison.json');assert cross['status']=='completed' and len(cross['results'])==10
    for r in cross['results']:
        rows=lines(data/'cross_domain_loss'/r['key']/'test_rows.jsonl');assert len(rows)==1000 and {x['id'] for x in rows}==test_ids['instruction_test_1000']
        loss_rows+=len(rows)
    assert loss_rows==69000
    statistical=read(report/'statistical_analysis.json')['primary'];assert len(statistical)==9
    for row in statistical:
        delta=scores[row['key']]-scores['reading_sft'];close(delta,row['em_interval']['delta_contender_minus_reference'])
        close(delta*2000,row['mcnemar']['gained_correct']-row['mcnemar']['lost_correct'])
    csv_files=sorted((report/'tables').glob('*.csv'));assert len(csv_files)==20
    counts={}
    for p in csv_files:
        with p.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream));assert rows and None not in rows[0]
            assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
        counts[p.name]=len(rows)
    assert counts['01_model_inventory.csv']==25 and counts['03_reading_all_checkpoints.csv']==22
    assert counts['08_reading_18000_paired_answers.csv']==18000 and counts['11_instruction_1000_all_models.csv']==15
    assert len(list((report/'figures').glob('*.png')))==9
    manifest=read(report/'report_manifest.json')
    for item in manifest['files']:
        path=report/item['path'];assert path.is_file() and path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    parser=ReportHTML();parser.feed((report/'完整测试报告.html').read_text(encoding='utf-8'))
    assert len(parser.ids)==len(set(parser.ids))
    assert all(href[1:] in parser.ids for href in parser.links if href.startswith('#'))
    assert not any(href.lower().startswith(('javascript:','data:')) for href in parser.links)
    assert len(parser.images)==9
    for image in parser.images:
        assert image.startswith('data:image/png;base64,')
        assert base64.b64decode(image.split(',',1)[1]).startswith(b'\x89PNG\r\n\x1a\n')
    cases=read(report/'case_studies.json');assert len(cases)==23
    for name in ['SFT测试报告.md','LoRA测试报告.md','RL测试报告.md']:
        content=(report/name).read_text(encoding='utf-8');assert len(content)>20000
    result={'status':'verified','generated_rows':generated,'teacher_forced_rows':loss_rows,'distinct_test_ids':3000,
        'reading_checkpoint_entries':22,'instruction_model_entries':15,'csv_tables':counts,'figures':9,
        'case_studies':len(cases),'sha256_checked_files':len(manifest['files']),'html_anchors_unique':True,'html_images_embedded':9,
        'reading_sft_em':scores['reading_sft'],'grpo_old_full_best_em':scores['grpo_old_full_50k_best']}
    (work/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True,type=Path);verify(p.parse_args().work)
