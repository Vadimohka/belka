import hashlib,json
from pathlib import Path
import pytest
from data_pipeline.quality import language_problem,shingles,jaccard,pack_shingles,unpack_shingles
from data_pipeline.contracts import sha256_file
from data_pipeline.leakage import normalized
ROOT=Path(__file__).resolve().parents[1]

def benchmark_fixture(tmp_path):
    path=tmp_path/'benchmark.jsonl';text=' '.join('бенчмаркслова'+str(i) for i in range(20))
    rows=[dict(sentence=text,word='мова',instruction='Гэта беларуская мова і гісторыя, якую важна ведаць.'),dict(sentence='кароткі сказ'),dict(sentence='Дарагавата')]
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=tmp_path/'benchmark-manifest.json';manifest.write_text(json.dumps(dict(name='fixture',revision='fixture',files=[dict(path=str(path),sha256=sha256_file(path),rows=3,config='cola',split='validation',input_fields=['sentence'])])))
    return manifest,text


def test_benchmark_only_semantic_fields_with_exact_token_verification(tmp_path):
    from data_pipeline.benchmark_leakage import BenchmarkIndex
    manifest,text=benchmark_fixture(tmp_path);idx=BenchmarkIndex(manifest,bloom_bits=10)
    assert idx.match('Уступ. '+text.replace(' ','\n')+' Завяршэнне.')
    assert idx.match('мова') is None
    assert idx.match('Гэта беларуская мова і гісторыя, якую важна ведаць.') is None
    assert idx.match('Кароткі, сказ!')
    assert idx.match('Гэта кароткі сказ у доўгім аповедзе.') is None
    assert idx.match('Дарагавата!')
    assert idx.match('На маю думку, гэта дарагавата для такой рэчы.') is None
    assert idx.proof()['coverage']['short_inputs']==2
    p=tmp_path/'benchmark.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):idx.verify_unchanged()


def test_benchmark_missing_semantic_field_fails_even_with_valid_json_and_hash(tmp_path):
    from data_pipeline.benchmark_leakage import BenchmarkIndex
    data=tmp_path/'missing.jsonl';data.write_text(json.dumps({'instruction':'Гэта толькі абгортка, а сапраўдны сказ адсутнічае.'})+'\n')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':[dict(path=str(data),sha256=sha256_file(data),rows=1,config='cola',split='test',input_fields=['sentence'])]}))
    with pytest.raises(ValueError,match='missing or empty semantic field'):
        BenchmarkIndex(manifest,bloom_bits=10)


def test_parallel_map_does_not_prefetch_unbounded_results():
    from data_pipeline.parallel import bounded_map
    consumed=0
    def values():
        nonlocal consumed
        for i in range(100000):consumed+=1;yield i
    class Pool:
        def map(self,fn,window,chunksize):return list(map(fn,window))
    stream=bounded_map(Pool(),lambda x:x,values(),workers=2,chunksize=4)
    assert next(stream)==0
    assert consumed<=16
    assert list(__import__('itertools').islice(stream,15))==list(range(1,16))
    assert consumed==16
    assert next(stream)==16 and consumed==32

def test_web_shared_russian_words_are_not_belarusian_evidence():
    text='На улице была хорошая погода для прогулки около дома. Солнце светило над городом, птицы пели в саду, дети играли на площадке, родители отдыхали на скамейке.'
    assert language_problem(text,'fineweb2')=='no_discriminative_belarusian_evidence'
    assert language_problem('Гэта беларуская мова і гісторыя, якую важна ведаць.','fineweb2') is None


def test_russian_sales_tail_and_review_domains():
    from data_pipeline.quality import url_problem
    intro='Гэта беларуская кніга пра гісторыю нашага горада і яго жыхароў. '
    tail='В нашем магазине вы можете приобрести книги по самым низким ценам. Система скидок действует при покупке товаров в интернет магазине. Мы предлагаем широкий выбор изданий и доставку по всей стране. '
    assert language_problem(intro*3+tail*2,'fineweb2')=='russian_script_dominant_web_segments'
    assert url_problem('https://be.eferrit.com/article','fineweb2')=='domain_requires_quality_review'
    assert url_problem('https://novychas.by/tags/kultura','fineweb2')=='tag_or_search_index'
    assert url_problem('https://notbe.eferrit.com.example.org/article','fineweb2') is None


def test_be_header_cannot_admit_english_or_ukrainian_web_body():
    head='Гэта беларуская мова і гісторыя, якую важна ведаць. '
    english=head+('This is an English article about the history of a city and the people who lived there. '*30)
    ukrainian=head+('Це українська мова, яка є важливою для історії нашої країни та її мешканців. '*30)
    assert language_problem(english,'fineweb2')=='non_cyrillic_dominant_web_text'
    assert language_problem(ukrainian,'hplt_v2')=='ukrainian_marked_web_text'
    technical='Гэта беларускае апісанне працы праграмы. API прымае запыт у фармаце JSON і вяртае адказ карыстальніку. OAuth дазваляе бяспечна ўвайсці ў праграму. Пры памылцы трэба праверыць налады і паўтарыць запыт.'
    assert language_problem(technical,'fineweb2') is None
    # Curated Cyrillic sources may legitimately include long Latin citations.
    assert language_problem(english,'bewiki') is None


@pytest.mark.parametrize('offset',[0,1,20,39])
def test_long_spans_are_detected_across_reflow_and_prefixes(offset):
    import sqlite3
    from data_pipeline.long_spans import quarantine_cross_split_spans
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,hash TEXT,text TEXT,gid TEXT,source TEXT,origin TEXT); CREATE TABLE selected(gid TEXT PRIMARY KEY,split TEXT)')
    common=[f'агульнаеслова{i}' for i in range(79)]
    train='Пачатак '+'\n'.join(common)+' канец аповеду'
    val=' '.join([f'уступ{i}' for i in range(offset)]+common+['завяршэнне'])
    db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?)',[(1,'train',train,'t','fineweb2','{}'),(2,'val',val,'v','fineweb2','{}')])
    db.executemany('INSERT INTO selected VALUES(?,?)',[('t','train'),('v','val')])
    records=[];proof=quarantine_cross_split_spans(db,lambda *a,**kw:records.append((a,kw)),bloom_bits=10)
    assert proof['removed_val_rows']==1 and proof['remaining_cross_split_long_span_hits']==0
    assert db.execute('SELECT COUNT(*) FROM docs').fetchone()[0]==1
    assert records[0][1]['matched_words']==40


def test_rolling_hash_collision_cannot_remove_unmatched_text(monkeypatch):
    import sqlite3,numpy as np
    import data_pipeline.long_spans as spans
    monkeypatch.setattr(spans,'hashes',lambda words:np.zeros(max(0,len(words)-39),dtype=np.uint64))
    db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,hash TEXT,text TEXT,gid TEXT,source TEXT,origin TEXT); CREATE TABLE selected(gid TEXT PRIMARY KEY,split TEXT)')
    db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?)',[(i,str(i),' '.join(f'тэкст{i}слова{n}' for n in range(100)),str(i),'fineweb2','{}') for i in (1,2)])
    db.executemany('INSERT INTO selected VALUES(?,?)',[('1','train'),('2','val')])
    proof=spans.quarantine_cross_split_spans(db,lambda *a,**kw:None,bloom_bits=10)
    assert proof['removed_val_rows']==0


def test_minor_source_coverage_is_honest_after_real_span_quarantine():
    import sqlite3,collections
    from data_pipeline.long_spans import quarantine_cross_split_spans
    from data_pipeline.quality import validation_coverage
    db=sqlite3.connect(':memory:')
    db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,hash TEXT,text TEXT,gid TEXT,source TEXT,origin TEXT); CREATE TABLE selected(gid TEXT PRIMARY KEY,split TEXT)')
    common=' '.join(f'агульнаеслова{i}' for i in range(79))
    main=common+' '+' '.join(f'асноўнытэкст{i}' for i in range(70000))
    small=' '.join(f'асобнаякніга{i}' for i in range(40))
    rows=[(1,'main',main,'main','fineweb2','{}'),
          (2,'mainval',' '.join(f'чыстаяправерка{i}' for i in range(80)),'mainval','fineweb2','{}'),
          (3,'book',small,'book','books_clean_v2','{}'),
          (4,'bookval','Уступ '+common+' канец','bookval','books_clean_v2','{}')]
    db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?)',rows)
    db.executemany('INSERT INTO selected VALUES(?,?)',[('main','train'),('mainval','val'),('book','train'),('bookval','val')])
    proof=quarantine_cross_split_spans(db,lambda *a,**kw:None,bloom_bits=10)
    assert proof['removed_val_rows']==1
    splits={part:dict(rows=0,sources={}) for part in ('train','val')}
    for part,source,text in db.execute('SELECT s.split,d.source,d.text FROM docs d JOIN selected s ON d.gid=s.gid'):
        splits[part]['rows']+=1
        entry=splits[part]['sources'].setdefault(source,dict(rows=0,chars=0))
        entry['rows']+=1;entry['chars']+=len(text)
    coverage=validation_coverage(splits)
    assert coverage['status']=='PASS_WITH_MINOR_GAPS'
    assert coverage['source_counts_complete']
    assert coverage['train_only_minor_sources']==['books_clean_v2']
    assert coverage['per_source']['books_clean_v2']['val_rows']==0
    assert coverage['per_source']['fineweb2']['status']=='COVERED'
    assert db.execute("SELECT COUNT(*) FROM docs WHERE hash='bookval'").fetchone()[0]==0


@pytest.mark.parametrize('name,rows,chars',[('be_x_oldwiki',1,10),('books_clean_v2',10000,100),('bewikisource_full',1,2000)])
def test_major_or_required_source_cannot_hide_missing_validation(name,rows,chars):
    from data_pipeline.quality import validation_coverage
    splits={'train':dict(rows=rows+1,sources={'fineweb2':dict(rows=1,chars=999000),name:dict(rows=rows,chars=chars)}),
            'val':dict(rows=1,sources={'fineweb2':dict(rows=1,chars=1000)})}
    coverage=validation_coverage(splits)
    assert coverage['status']=='FAIL'
    assert coverage['missing_required_sources']==[name]
    assert not coverage['train_only_minor_sources']


@pytest.mark.parametrize('summary',[
    {'rows':1},
    {'rows':1,'sources':{}},
    {'rows':2,'sources':{'bewiki':{'rows':1,'chars':100}}},
    {'rows':1,'chars':101,'sources':{'bewiki':{'rows':1,'chars':100}}},
    {'rows':1,'sources':{'bewiki':{'rows':-1,'chars':100}}},
    {'rows':True,'sources':{'bewiki':{'rows':True,'chars':100}}},
    {'rows':1,'sources':{'bewiki':{'rows':1,'chars':float('nan')}}},
    {'rows':1,'sources':{'bewiki':{'rows':1,'chars':0}}},
])
def test_validation_coverage_requires_complete_valid_source_accounting(summary):
    from data_pipeline.quality import validation_coverage
    coverage=validation_coverage({'train':summary,'val':{'rows':1,'sources':{'bewiki':{'rows':1,'chars':100}}}})
    assert coverage['status']=='FAIL'
    assert not coverage['source_counts_complete']
    assert coverage['source_count_errors']


def test_sketch_requires_actual_high_jaccard_and_is_deterministic():
    text=' '.join('слова'+str(i) for i in range(200));variant=text.replace('слова150 ','замена ')
    a,b=shingles(text),shingles(variant)
    assert .85<jaccard(a,b)<1
    assert set(sorted(a)[:4])&set(sorted(b)[:4])
    assert unpack_shingles(pack_shingles(a))==a
    assert jaccard(a,shingles(' '.join('іншае'+str(i) for i in range(200))))<.01


@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('coverage_failure,alias',[(False,None),(True,None),(False,'bewikisource_full'),(False,'bewikibooks_full')])
def test_finalize_removes_verified_near_foreign_and_long_paragraph_leakage(tmp_path,workers,coverage_failure,alias):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    from tools.finalize_h200_data import finalize
    from tools.build_sft_mix import build_v9
    from types import SimpleNamespace
    inp=tmp_path/'input';inp.mkdir();(inp/'quarantine.jsonl').write_text('');rows=[];benchmark,benchmark_text=benchmark_fixture(tmp_path)
    head='Гэта беларуская мова і культура, якую важна ведаць. '
    common=head+' '.join('агульны'+str(i) for i in range(60))
    for i in range(16):
        text=head+' '.join(f'тэкст{i}слова{j}' for j in range(200))
        if i in (1,2) and not coverage_failure:text+='\n'+common
        if coverage_failure:text+=' '+' '.join(f'супольнаеслова{j}' for j in range(79))
        rows.append(dict(text=text,source='fineweb2',group_id='g'+str(i),orthography='unspecified',chars=len(text),content_sha256=hashlib.sha256(normalized(text).encode()).hexdigest(),input_file='fixture',input_row=i,source_doc_id=None))
    near=dict(rows[0]);near['text']=near['text'].replace('тэкст0слова150 ','замена ');near['content_sha256']=hashlib.sha256(normalized(near['text']).encode()).hexdigest();near['group_id']='near';rows.append(near)
    bad=dict(rows[0]);bad['text']='На улице была хорошая погода для прогулки около дома. Солнце светило над городом, птицы пели в саду, дети играли на площадке, родители отдыхали на скамейке.';bad['group_id']='ru';bad['content_sha256']=hashlib.sha256(normalized(bad['text']).encode()).hexdigest();rows.append(bad)
    contaminated=dict(rows[0]);contaminated['text']+=' '+benchmark_text;contaminated['group_id']='benchmark';contaminated['content_sha256']=hashlib.sha256(normalized(contaminated['text']).encode()).hexdigest();rows.append(contaminated)
    if alias:
        # Its original paragraph passes the source's admission minimum, but
        # dedup leaves a 30..199-char fragment. Aliases must retain their actual
        # Wikisource/Wikibooks minima (400/200), not the fallback minimum of 30.
        tail='Гэта асобная кароткая беларуская заўвага пра гісторыю кнігі.'
        assert 30<len(tail)<200
        fragment=dict(rows[0],source=alias,group_id='alias-fragment',text=common+'\n'+tail)
        fragment['chars']=len(fragment['text']);fragment['content_sha256']=hashlib.sha256(normalized(fragment['text']).encode()).hexdigest();rows.append(fragment)
    files=[]
    for split,rs in [('train',rows[:12]),('val',rows[12:])]:
        p=inp/(split+'_00000.parquet');pq.write_table(pa.Table.from_pylist(rs),p)
        files.append(dict(path=p.name,split=split,rows=len(rs),sha256=sha256_file(p)))
    out=tmp_path/'output';out.mkdir();build_v9(SimpleNamespace(train_out=None,val_out=None,check_only=False),ROOT,out);sf=(out/'.sft_current').resolve();holdout=ROOT/'eval/strict_holdout_quality_control_v2.be.jsonl'
    report=dict(schema='belka-h200-data-v1',status='READY_FOR_TOKENIZER',corpus_dir=str(inp),files=files,corpus_fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_policy_sha256=sha256_file(ROOT/'configs/source_mixing_policy.yaml'),splits={'train':{'rows':12},'val':{'rows':len(rows)-12}},split_policy={'val_ratio':.2},counts={'seen':len(rows)},quarantine={'rows':0,'file':'quarantine.jsonl','sha256':sha256_file(inp/'quarantine.jsonl')},holdout={'path':str(holdout),'sha256':sha256_file(holdout)},sft={'files':[{'path':str(sf/n),'sha256':sha256_file(sf/n)} for n in ('identity_conversations.jsonl','identity_conversations_val.jsonl')]})
    proof=tmp_path/'input.json';proof.write_text(json.dumps(report))
    if coverage_failure:
        with pytest.raises(ValueError,match='emptied a split'):
            finalize(proof,out,tmp_path/'final.json',workers=workers,benchmark_manifest=benchmark)
        failure=json.loads((tmp_path/'final.failed.json').read_text())
        assert failure['status']=='FAILED' and failure['output_published'] is False
        assert failure['splits']['val']['rows']==0
        assert failure['validation_coverage']['missing_required_sources']==['fineweb2']
        assert failure['counts']['cross_split_span_quarantine_rows']>0
        assert not (out/'.corpus_current').exists()
        assert all(sha256_file(inp/e['path'])==e['sha256'] for e in files)
        return
    final=finalize(proof,out,tmp_path/'final.json',workers=workers,benchmark_manifest=benchmark)
    q=final['quality_pass'];assert q['status']=='PASS' and q['all_checks_complete']
    assert q['counts']['language_quarantine_rows']==1
    assert q['near_dedup']['removed_rows']==1
    assert q['paragraph_dedup']['removed_occurrences']==1+bool(alias)
    assert q['paragraph_dedup']['cross_split_duplicate_paragraphs']==0
    assert q['final_rows_verified']==16
    assert q['validation_coverage']['status']=='PASS'
    assert final['benchmark_decontamination']['removed_corpus_rows']==1
    assert final['benchmark_decontamination']['remaining_corpus_hits']==0
    assert final['counts']['seen']==final['counts']['final_rows']+final['counts']['quality_removed_rows']
    if alias:
        assert q['counts']['post_trim_quarantine_rows']==1
        assert q['decisions_by_source'][alias]['empty_or_short_after_paragraph_dedup']==1
