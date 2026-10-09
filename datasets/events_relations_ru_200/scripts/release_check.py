import json,collections
from pathlib import Path
from cases import CASES
from build_batch import rows
rr=[r for i in range(len(CASES)) for r in rows(i)]
assert len(rr)==200 and len({r['id'] for r in rr})==200
splits={k:{r['metadata']['family_id'] for r in rr if r['metadata']['split']==k} for k in ['train','validation','test']}
assert all(not splits[a]&splits[b] for a,b in [('train','test'),('train','validation'),('test','validation')])
for i in range(len(CASES)):
    rs=rr[i*5:i*5+5];basic=json.loads(rs[0]['output']);ev=json.loads(rs[1]['output']);graph=json.loads(rs[2]['output']);groups=json.loads(rs[3]['output']);chain=json.loads(rs[4]['output'])
    assert len(basic)==len(ev)==len(graph['relations'])==sum(len(g['actions']) for g in groups)==len(chain['events'])
    participants={n['id'] for n in graph['participants']}
    for r in graph['relations']:
        assert set(r['actors'])<=participants
        assert r['target'] is None or r['target'] in participants
    vids={v['id'] for v in chain['events']}
    for link in chain['links']:assert link['from'] in vids and link['to'] in vids
report=dict(examples=200,independent_texts=40,views=5,events=sum(len(c['events']) for c in CASES),
 explicit_event_links=sum(len(c['links']) for c in CASES),unknown_actor_events=sum(e['actor'] is None for c in CASES for e in c['events']),
 split_rows=dict(collections.Counter(r['metadata']['split'] for r in rr)),
 input_tokens_mt5=dict(min=min(r['metadata']['input_tokens_mt5'] for r in rr),max=max(r['metadata']['input_tokens_mt5'] for r in rr)),
 checks=['json_schema','participants_references','event_references','view_event_counts','split_disjoint','input_le_2048_mt5'],
 provenance='assistant_authored_synthetic',independent_human_review=False,training_improvement_measured=False)
Path(__file__).with_name('quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
