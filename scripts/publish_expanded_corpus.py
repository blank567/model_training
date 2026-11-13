"""校验并发布原 Wiki + 增量网页语料；只在完整验收后产生 manifest.json。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from llm_lab.common.io import read_json, sha256_file, write_json
from llm_lab.data.common import ensure_under, json_line, read_jsonl, text_hash
from llm_lab.tokenization.bpe import load_tokenizer


def hash_segment(path, start=0, size=None):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        handle.seek(start)
        while size is None or size:
            chunk=handle.read(4*1024**2 if size is None else min(size,4*1024**2))
            if not chunk:
                if size:raise ValueError('Truncated segment')
                break
            digest.update(chunk)
            if size is not None:size-=len(chunk)
    return digest.hexdigest()


def verified_link(source, destination, digest):
    if source.is_symlink() or sha256_file(source)!=digest:raise ValueError('Parent file identity changed')
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination)!=digest:raise ValueError('Publication target mismatch')
    else:
        os.link(source,destination)


def publish(output: str | Path) -> Path:
    output=Path(output).resolve()
    if (output/'writer.lock').exists():raise RuntimeError('Expansion writer still running')
    marker=output/'publish.lock'
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(fd)
    try:
        expansion_path=output/'expansion_manifest.json'
        expansion=read_json(expansion_path)
        parent_path=Path(expansion['parent_manifest']['path']).resolve()
        if sha256_file(parent_path)!=expansion['parent_manifest']['sha256']:raise ValueError('Parent manifest changed')
        parent=read_json(parent_path)
        audit_path=ensure_under(output/expansion['audit']['path'],output)
        if sha256_file(audit_path)!=expansion['audit']['sha256']:raise ValueError('Incremental audit changed')
        for name in ('audit','documents','tokenizer'):
            p=ensure_under(parent_path.parent/parent[name]['path'],parent_path.parent)
            if sha256_file(p)!=parent[name]['sha256']:raise ValueError('Parent '+name+' changed')
        result_path=output/'manifest.json'
        if result_path.exists():
            old=read_json(result_path)
            if old.get('expansion_manifest',{}).get('sha256')!=sha256_file(expansion_path):raise ValueError('Publication identity changed')
            for split,languages in old['splits'].items():
                for item in languages.values():
                    if sha256_file(ensure_under(output/item['path'],output))!=item['sha256']:raise ValueError('Published token stream changed')
            return result_path
        tokenizer=load_tokenizer(parent_path.parent/parent['tokenizer']['path'])
        vocab,eos=parent['tokenizer']['vocab_size'],parent['tokenizer']['eos_id']
        streams={}
        segments={}
        arrays={}
        for lang in ('en','zh'):
            item=expansion['splits']['train'][lang]
            path=ensure_under(output/item['path'],output)
            base=parent['splits']['train'][lang]
            count=expansion['base_prefix'][lang]['token_count']
            if count!=base['token_count'] or item['token_count']!=count+expansion['new_tokens_per_language'][lang]:raise ValueError('Prefix/count mismatch')
            if path.stat().st_size!=item['token_count']*4 or sha256_file(path)!=item['sha256']:raise ValueError('Train stream mismatch')
            if hash_segment(path,0,count*4)!=base['sha256']:raise ValueError('Original training prefix changed')
            streams[lang]={**item,'byte_order':'little'}
            segments[lang]={'base_tokens':count,'base_sha256':base['sha256'],
                            'new_tokens':expansion['new_tokens_per_language'][lang],
                            'new_segment_sha256':hash_segment(path,count*4)}
            arrays[lang]=np.memmap(path,dtype='<u4',mode='r')
        # 独立审计每篇偏移、EOS 与词表；抽样解码后校验正文 hash，防止计数或错位。
        offsets={lang:segments[lang]['base_tokens'] for lang in ('en','zh')}
        documents={'en':0,'zh':0}
        samples={'en':0,'zh':0}
        scratch=output/'publication_checks.sqlite'
        if scratch.exists():scratch.unlink()
        checks=sqlite3.connect(scratch)
        checks.execute('CREATE TABLE unique_docs(hash TEXT PRIMARY KEY,source TEXT UNIQUE)')
        merged=output/'document_audit.jsonl.tmp'
        with merged.open('wb') as writer:
            with (parent_path.parent/parent['audit']['path']).open('rb') as reader:
                for chunk in iter(lambda:reader.read(1024**2),b''):writer.write(chunk)
            for row in read_jsonl(audit_path):
                if row.get('status')!='encoded':continue
                lang=row['language'];count=row['token_count'];offset=row['token_offset']
                if offset!=offsets[lang] or offset+count>len(arrays[lang]) or count<1:raise ValueError('Audit token gap/overlap')
                ids=arrays[lang][offset:offset+count]
                if int(ids[-1])!=eos or int(ids.max())>=vocab:raise ValueError('EOS/vocab mismatch')
                if documents[lang]%5000==0:
                    decoded=tokenizer.decode(ids[:-1].tolist(),skip_special_tokens=False)
                    if text_hash(decoded)!=row['text_sha256']:raise ValueError('Decoded text/hash mismatch')
                    samples[lang]+=1
                checks.execute('INSERT INTO unique_docs VALUES (?,?)',(row['text_sha256'],row['source_id']))
                offsets[lang]+=count;documents[lang]+=1
                writer.write(json_line(row).encode('utf-8'))
            writer.flush();os.fsync(writer.fileno())
        checks.close();scratch.unlink()
        if any(offsets[lang]!=streams[lang]['token_count'] for lang in ('en','zh')):raise ValueError('Audit does not cover all appended tokens')
        if sum(documents.values())!=expansion['encoded_documents']:raise ValueError('Document count mismatch')
        os.replace(merged,output/'document_audit.jsonl')
        splits={'train':streams}
        for split in ('validation','test'):
            splits[split]={}
            for lang in ('en','zh'):
                item=parent['splits'][split][lang];name=f'{split}.{lang}.bin'
                verified_link(parent_path.parent/item['path'],output/name,item['sha256'])
                splits[split][lang]={**item,'path':name}
        for key,name in (('tokenizer','tokenizer.json'),('documents','documents.jsonl')):
            item=parent[key]
            verified_link(parent_path.parent/item['path'],output/name,item['sha256'])
        result={**expansion,'kind':'pretrain_token_streams','preparation':'incremental_web_expansion_v1',
                'tokenizer':{**parent['tokenizer'],'path':'tokenizer.json'},'splits':splits,
                'token_count':sum(x['token_count'] for languages in splits.values() for x in languages.values()),
                'audit':{'path':'document_audit.jsonl','sha256':sha256_file(output/'document_audit.jsonl'),'role':'isolation_ledger'},
                'documents':{**parent['documents'],'path':'documents.jsonl'},
                'incremental_audit':expansion['audit'],
                'expansion_manifest':{'path':expansion_path.name,'sha256':sha256_file(expansion_path)},
                'segments':segments,
                'sampling':{'unit':'effective_tokens','language_probabilities':{'en':.5,'zh':.5},'note':'随机有放回窗口采样；语料容量不等于完整覆盖量。'},
                'acceptance':{'status':'passed','audit_tokens_contiguous':True,'audit_documents':documents,'decoded_hash_samples':samples,
                              'parent_train_prefixes_unchanged':True,'parent_holdout_streams_unchanged':True},
                'note':'旧 Wikipedia 约1.1B + 新网页约2B；简体中文/英文各半，冻结 tokenizer 与旧 holdout；精确及100字符片段去重不能证明所有语义近重复被移除。'}
        write_json(output/'acceptance.json',result['acceptance'])
        write_json(result_path,result)
        return result_path
    finally:
        marker.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(publish(parser.parse_args().output))
