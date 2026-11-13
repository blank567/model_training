"""规范化题目/段落/文章标识，并以字符shingle筛查近重复。"""
from collections import defaultdict
from pathlib import Path
import heapq,json,re,unicodedata,zlib
from urllib.parse import unquote,urlsplit
from llm_lab.common.io import read_json,sha256_file,canonical_hash


def compact(text):
    return re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',text).casefold())


def article_names(title='',url=''):
    names=set()
    if title.strip():
        names.add(compact(title.replace('_',' ')))
        cleaned=re.sub(r'[_｜|]\s*(?:百度百科|维基百科|Wikipedia).*$', '',title,flags=re.I)
        names.add(compact(cleaned))
    try:parts=urlsplit(url or '');host=parts.hostname or ''
    except ValueError:parts=urlsplit('');host=''
    if host=='wikipedia.org' or host.endswith('.wikipedia.org'):
        path=unquote(parts.path)
        if '/wiki/' in path:names.add(compact(path.split('/wiki/',1)[1].replace('_',' ')))
    return names-{''}


def shingles(text):
    text=compact(text)
    return {zlib.crc32(text[i:i+13].encode('utf-8')) for i in range(max(0,len(text)-12))}


def anchors(text):
    clean=compact(text);values=[zlib.crc32(clean[i:i+13].encode('utf-8')) for i in range(max(0,len(clean)-12))]
    if not values:return set()
    return {min(values[i:i+8]) for i in range(max(1,len(values)-7))}


class IsolationIndex:
    def __init__(self):
        self.questions=set();self.passages=set();self.articles=set();self.features=[];self.anchors=defaultdict(list)
        self.bound_sources=[]

    def add_passage(self,text):
        clean=compact(text)
        if len(clean)<30 or clean in self.passages:return
        self.passages.add(clean);values=shingles(clean);index=len(self.features);self.features.append(values)
        for key in anchors(clean):self.anchors[key].append(index)

    def add_row(self,row):
        self.questions.add(compact(row.get('question') or row.get('prompt','')))
        self.articles.update(article_names(row.get('title','')))
        self.add_passage(row.get('context') or row.get('input_context',''))

    def reason(self,question,context,title='',url=''):
        if compact(question) in self.questions:return 'sft_or_evaluation_question_overlap'
        if self.articles & article_names(title,url):return 'sft_or_evaluation_article_overlap'
        clean=compact(context)
        if clean in self.passages:return 'sft_or_evaluation_passage_overlap'
        values=shingles(clean)
        if len(values)<30:return None
        candidates=set()
        for key in anchors(clean):candidates.update(self.anchors.get(key,()))
        for index in candidates:
            other=self.features[index];overlap=len(values&other)
            if overlap>=30 and overlap/min(len(values),len(other))>=.85:return 'sft_or_evaluation_near_passage'
        return None


def exclusion_index(manifests):
    index=IsolationIndex()
    for path in manifests:
        path=Path(path);manifest=read_json(path);splits={}
        for split in ['train','validation','test']:
            spec=manifest['splits'][split];file=path.parent/spec['path']
            if sha256_file(file)!=spec['sha256']:raise ValueError('Exclusion corpus changed')
            for line in file.open(encoding='utf-8'):index.add_row(json.loads(line))
            splits[split]={'examples':spec['examples'],'sha256':spec['sha256']}
        index.bound_sources.append({'path':str(path),'manifest_sha256':sha256_file(path),
                                    'data_identity':manifest.get('data_identity'),'splits':splits})
    return index


def export_reference(index,path):
    from llm_lab.common.io import write_json
    write_json(path,{'schema_version':1,'questions':sorted(index.questions),'passages':sorted(index.passages),
        'articles':sorted(index.articles),'source_manifests':index.bound_sources})


def load_reference(path):
    data=read_json(path);index=IsolationIndex();index.questions=set(data['questions']);index.articles=set(data['articles'])
    for passage in data['passages']:index.add_passage(passage)
    index.bound_sources=data['source_manifests'];return index
