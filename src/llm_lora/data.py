"""只读辅助开发/最终测试协议，沿用已验证的错位和Mask。"""
from pathlib import Path
import json

from llm_lab.common.io import canonical_hash,read_json,sha256_file
from llm_lab.tokenization.bpe import load_tokenizer
from llm_sft.data import SFTCorpus


def contained(directory,name):
    path=Path(directory)/name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError('Evaluation path escaped its directory')
    return path


class AuxiliaryCorpus(SFTCorpus):
    def __init__(self,manifest_path,split,expected_sha256,*,allow_fixture=False):
        if split not in {'validation','test'}:raise ValueError('Auxiliary data is evaluation-only')
        raw=Path(manifest_path)
        if raw.is_symlink() or sha256_file(raw)!=expected_sha256:raise ValueError('Evaluation manifest changed')
        self.manifest_path=raw.resolve();self.directory=self.manifest_path.parent;self.manifest=read_json(raw)
        manifest=self.manifest
        if manifest.get('kind')!='sft_auxiliary_evaluation':raise ValueError('Unexpected evaluation protocol')
        self.fixture=manifest.get('fixture') is True
        if self.fixture and not allow_fixture:raise ValueError('Fixture evaluation disabled')
        acceptance=read_json(self.directory/'acceptance.json')
        if acceptance.get('status')!='passed' or not acceptance.get('training_review_and_split_isolation'):
            raise ValueError('Evaluation isolation acceptance missing')
        self.split=split;spec=manifest['files'][split];path=contained(self.directory,spec['path'])
        if sha256_file(path)!=spec['sha256']:raise ValueError('Evaluation split changed')
        token=manifest['tokenizer'];self.tokenizer_path=contained(self.directory,token['path'])
        if sha256_file(self.tokenizer_path)!=token['sha256']:raise ValueError('Tokenizer changed')
        self.pad_id=token['pad_id'];self.unk_id=token['unk_id'];self.eos_id=token['eos_id'];self.vocab_size=token['vocab_size']
        self.max_sequence_tokens=manifest['tokenization']['max_sequence_tokens']
        tokenizer=load_tokenizer(self.tokenizer_path);self.rows=[];self.language_indices={'en':[],'zh':[]};seen=set()
        with path.open(encoding='utf-8') as stream:
            for number,line in enumerate(stream,1):
                row=json.loads(line);self._validate_row(row,number)
                if row['id'] in seen:raise ValueError('Duplicate evaluation ID')
                seen.add(row['id'])
                template=manifest['tokenization']['prefix_template' if row['system'] else 'empty_system_prefix_template']
                prefix=tokenizer.encode(template.format(**row),add_special_tokens=False).ids
                answer=tokenizer.encode(row['response'],add_special_tokens=False).ids+[self.eos_id]
                if prefix+answer!=row['input_ids'] or row['loss_mask']!=[0]*len(prefix)+[1]*len(answer):
                    raise ValueError('Evaluation text/Token/Mask mismatch')
                self.language_indices[row['language']].append(len(self.rows));self.rows.append(row)
        if len(self.rows)!=spec['examples'] or not all(self.language_indices.values()):raise ValueError('Evaluation count or language mismatch')
        self.identity={'manifest_sha256':expected_sha256,'split':split,'split_sha256':spec['sha256'],
                       'tokenizer_sha256':token['sha256'],'examples':len(self.rows)}


def protocol(config):
    path=config.get('evaluation_protocol')
    if not path:return None
    if sha256_file(path)!=config.get('evaluation_protocol_sha256'):raise ValueError('LoRA evaluation protocol changed')
    value=read_json(path)
    if value.get('kind')!='lora_evaluation_protocol' or value.get('status')!='frozen_before_lora_tuning':
        raise ValueError('Evaluation protocol was not frozen')
    return value


def development(config):
    spec=protocol(config)
    if spec is None:return None
    source=spec['development']
    return AuxiliaryCorpus(source['manifest'],'validation',source['manifest_sha256'],allow_fixture=config['fixture'])
