import json,argparse,os,statistics
from pathlib import Path
import sentencepiece as spm
from jsonschema import Draft202012Validator
from cases import CASES
W=Path(__file__).parent
ROOT='datasets/events_relations_ru_200'
sp=spm.SentencePieceProcessor(model_file=os.environ.get('T5SET_SPIECE_MODEL',str(W.parents[1]/'meetings_hard_context_work/spiece.model')))
def compact(o):return json.dumps(o,ensure_ascii=False,separators=(',',':'))
def ob(p,d):return dict(description=d,type='object',properties=p,required=list(p),additionalProperties=False)
def ar(x):return dict(type='array',items=x)
st={'type':'string'};nullst={'type':['string','null']}
actors={'type':'array','items':st}
def rows(i):
    c=CASES[i];names=c['participants'];assert len(names)==len(set(names))
    def actorlist(e):return e['actor'] if isinstance(e['actor'],list) else ([] if e['actor'] is None else [e['actor']])
    for event in c['events']:
        assert set(actorlist(event))<=set(names)
        assert all(event[k] is None or event[k] in names for k in ['target','recipient','instrument','location','source'])
    rule='Извлеки события, о совершении которых текст явно утверждает. Разрешай местоимения и прозвища только по достаточному контексту. Не включай отрицательные действия, одни намерения и неслучившиеся условия. Действие в инфинитиве; не подменять попытку успешным результатом. actor: явно определённые исполнители; совместное действие — одна запись со всеми исполнителями. Если исполнитель неоднозначен, actor=[]. Не добавляй подразумеваемых действий. Сохраняй порядок первого описания событий.'
    basic=ob({'actor':actors,'action':st,'target':nullst},'Один факт действия. target — объект действия, адресат помощи/вызова или место назначения входа. У передачи target — передаваемый предмет. Неизвестный или отсутствующий объект null.')
    evt=ob({'actor':actors,'action':st,'target':nullst,'recipient':nullst,'instrument':nullst,'location':nullst,'source':nullst},'Одно событие с ролями. recipient — получатель передачи, instrument — использованное средство, location — место, source — названный отправитель при получении или источник перемещения. Неназванная роль null. Не выводить новые действия из ролей.')
    triple=ar(basic);triple['description']=rule+' Только исполнитель, действие и объект; не превращать получателя предмета в прямой объект передачи.'
    events=ar(evt);events['description']=rule+' Сохраняй отдельно объект, получателя, средство и место; для получения recipient не означает отправителя.'
    idst={'type':'string','pattern':'^p[1-9][0-9]*$'}
    graph=ob({'participants':ar(ob({'id':idst,'name':st},'Именованный участник, объект, средство или место; одна сущность — один ID.')),
      'relations':ar(ob({'actors':ar(idst),'action':st,'target':{'type':['string','null'],'pattern':'^p[1-9][0-9]*$'}},'Действие направлено от исполнителей к объекту.'))},
      rule+' participants: все явно упомянутые именованные сущности, даже без фактических действий; p1,p2,... по первому упоминанию. relations: ссылки на эти ID. Получатель передачи — отдельная роль, не прямой объект; здесь эту роль не выводить.')
    grouped=ar(ob({'actors':actors,'actions':ar(ob({'action':st,'target':nullst},'Действие группы исполнителей.'))},'Одинаковую группу исполнителей объединить; совместных исполнителей не разъединять.'))
    grouped['description']=rule+' Сгруппировать события по точному набору исполнителей в порядке первого события группы.'
    chain=ob({'events':ar(ob({'id':{'type':'string','pattern':'^v[1-9][0-9]*$'},'actor':actors,'action':st,'target':nullst},'ID v1,v2,... по порядку событий.')),
      'links':ar(ob({'from':st,'relation':{'type':'string','enum':['before','causes','enables']},'to':st},'Только прямо выраженная связь событий.'))},
      rule+' links: before — явное «потом/после/затем/прежде»; causes — прямо заявлено, что событие вызвало другое; enables — прямо заявлено, что событие сделало другое возможным. Не создавать порядок лишь из порядка предложений и не считать «после» причинностью. Не добавлять транзитивные связи.')
    schemas=dict(action_relations=triple,event_roles=events,participant_graph=graph,by_actor=grouped,event_chain=chain)
    basics=[dict(actor=actorlist(x),action=x['action'],target=x['target']) for x in c['events']]
    mapped={name:f'p{j+1}' for j,name in enumerate(names)}
    groups={}
    for b in basics:groups.setdefault(tuple(b['actor']),[]).append(dict(action=b['action'],target=b['target']))
    outputs=dict(action_relations=basics,event_roles=[dict(x,actor=actorlist(x)) for x in c['events']],
      participant_graph=dict(participants=[dict(id=mapped[n],name=n) for n in names],relations=[dict(actors=[mapped[n] for n in b['actor']],action=b['action'],target=mapped[b['target']] if b['target'] else None) for b in basics]),
      by_actor=[dict(actors=list(k),actions=v) for k,v in groups.items()],
      event_chain=dict(events=[dict(id=f'v{j+1}',**b) for j,b in enumerate(basics)],links=[dict(**{'from':f'v{a+1}'},relation=r,to=f'v{b+1}') for a,r,b in c['links']]))
    for a,r,b in c['links']:assert 0<=a<len(basics) and 0<=b<len(basics) and a!=b
    rr=[]
    for view,sch in schemas.items():
        Draft202012Validator.check_schema(sch);Draft202012Validator(sch).validate(outputs[view])
        inp=f'<schema>{compact(sch)}</schema>\n<text>{c["text"]}</text>';tok=len(sp.encode(inp))+1
        assert tok<=2048,(i,view,tok)
        rr.append(dict(id=f'a{i+1:02}_{view}',input=inp,output=compact(outputs[view]),metadata=dict(family_id=f'a{i+1:02}',scenario=c['topic'],view=view,split='train' if i<32 else ('validation' if i<36 else 'test'),origin='assistant_authored_synthetic',input_tokens_mt5=tok,tokenizer='google/mt5-small@c83db3e90edd4c0bc69ad9f075c3d19b1ef56c45')))
    return rr
def readme(n):return f'''# Кто → что сделал → с кем или чем

Опубликовано {n} из 200 пар: план — 40 самостоятельных текстов × 5 представлений. Это событийные связи, а не карточки персонажей.

Пример: «Дед посадил репку. Потом он позвал бабку» → `дед — посадить → репка`, `дед — позвать → бабка`.

Представления: `action_relations` — исполнитель/действие/объект; `event_roles` — плюс получатель, средство, место; `participant_graph` — ID участников и направленные действия; `by_actor` — группировка действий; `event_chain` — события и прямо выраженные связи before/causes/enables. Совместное действие сохраняется единым событием. Отрицание, намерение и неслучившееся условие не дают положительного факта. Неоднозначный исполнитель — [], неизвестная роль — null. Попытка не становится успешным действием.

Формат каждой JSONL-строки: `input` со схемой и текстом, `output` — строка с JSON, `metadata` — происхождение, семейство, split. Склеивайте `batch_*.jsonl` построчно. 32 семейства train, 4 validation, 4 test; все представления одного текста в одной части.

Русские тексты и разметка синтетические, написаны ассистентом; независимого human gold нет. Проверяются JSON Schema, ссылки на участников/события и длина полного входа по указанному mT5. Не добавляются обратные или транзитивные связи, порядок предложений не считается причинностью. Для своего checkpoint перепроверьте длины. Польза для обучения пока не измерена. Это короткие и средние тексты, не полноразмерные стенограммы.
'''
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('batch',type=int);a=p.parse_args();start=(a.batch-1)*2
    rr=rows(start)+rows(start+1);content=''.join(compact(x)+'\n' for x in rr)
    (W/f'batch_{a.batch:03}.jsonl').write_text(content)
    print(compact(dict(entries=[dict(path=f'{ROOT}/batch_{a.batch:03}.jsonl',mode='100644',type='blob',content=content),dict(path=f'{ROOT}/README.md',mode='100644',type='blob',content=readme(a.batch*10))],rows=10,max_tokens=max(x['metadata']['input_tokens_mt5'] for x in rr))))
