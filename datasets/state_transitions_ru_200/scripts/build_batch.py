import json,os,sys,hashlib
from pathlib import Path
import sentencepiece as spm
from jsonschema import Draft202012Validator
from cases import CASES
W=Path(__file__).parent
ROOT='datasets/state_transitions_ru_200'
sp=spm.SentencePieceProcessor(model_file=os.environ.get('T5SET_SPIECE_MODEL',str(W.parents[1]/'meetings_hard_context_work/spiece.model')))
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def ob(p,d):return dict(type='object',description=d,properties=p,required=list(p),additionalProperties=False)
def ar(x,d=''):return dict(type='array',description=d,items=x)
st={'type':'string'}; val={'type':['string','null']}
RULE='Отслеживай только перечисленные ключи состояния. Ключ имеет вид «объект / поле», последний разделитель отделяет поле. Значения сохраняй в форме текста; однозначные местоимения и прозвища нормализуй к полному имени. null означает неизвестное значение, а не отсутствие человека или предмета. Начальные состояния — явно названные состояния до событий; если не названы, null. Учитывай только подтверждённые изменения: не намерения, предложения, отрицания, неисполненные условия, неудачные попытки и не связанные с ключами действия. Исправленное ошибочное сообщение замени подтверждённым фактом; не считать его отдельным реальным переходом. Упорядочи переходы по явно указанному времени, иначе по явно описанной хронологии. Одновременные изменения перечисляй в порядке ключей. Сравнение итогов с началом не подменяет историю. Не дописывай события из общих знаний.'
def rows(i):
 c=CASES[i];state=dict(c['initial']);history=[]
 assert set(c['initial'])==set(c['final'])
 for n,e in enumerate(c['events'],1):
  assert e['key'] in state and e['evidence'] in c['text']
  assert state[e['key']]!=e['value'],(i,e)
  history.append(dict(id=f'e{n}',key=e['key'],before=state[e['key']],after=e['value'],evidence=e['evidence']))
  state[e['key']]=e['value']
 assert state==c['final'],(i,state,c['final'])
 scope=RULE+' Ключи и порядок: '+compact(list(state))+'.'
 item=ob({'key':st,'value':val},'Состояние одного отслеживаемого поля.')
 event=ob({'id':st,'key':st,'before':val,'after':val,'evidence':st},'Подтверждённый переход; e1,e2,... по хронологии. evidence — дословный фрагмент текста, достаточный для нового значения; для раскрытия местоимения используй остальной текст.')
 change=ob({'key':st,'before':val,'after':val},'Начальное и конечное значения поля с различными значениями. Промежуточные переходы здесь не перечислять.')
 group=ob({'key':st,'initial':val,'transitions':ar(ob({'before':val,'after':val,'evidence':st},'Переход этого поля в хронологическом порядке.')),'current':val},'История одного поля, включая поля без изменений.')
 schemas={
  'current_state':ar(item,scope+' Верни состояние на конец текста, все ключи в указанном порядке.'),
  'timeline':ob({'initial':ar(item),'events':ar(event),'current':ar(item)},scope+' Начальные состояния, полная хронология реальных переходов и итог; все состояния в порядке ключей.'),
  'net_changes':ar(change,scope+' Сравни начальное и конечное состояние. Верни только ключи с различными значениями, в порядке ключей. Если значение вернулось к начальному, не включай ключ. При неизвестном начале before=null.'),
  'field_histories':ar(group,scope+' Сгруппируй переходы по ключу, все ключи в указанном порядке; transitions=[] при отсутствии переходов.')}
 outputs={
  'current_state':[dict(key=k,value=v) for k,v in state.items()],
  'timeline':dict(initial=[dict(key=k,value=v) for k,v in c['initial'].items()],events=history,current=[dict(key=k,value=v) for k,v in state.items()]),
  'net_changes':[dict(key=k,before=c['initial'][k],after=v) for k,v in state.items() if c['initial'][k]!=v],
  'field_histories':[dict(key=k,initial=c['initial'][k],transitions=[{x:e[x] for x in ['before','after','evidence']} for e in history if e['key']==k],current=v) for k,v in state.items()]}
 out=[]
 # Hold out whole sources, spread across domains rather than tail-only topics.
 split='validation' if i%10==8 else ('test' if i%10==9 else 'train')
 for view,schema in schemas.items():
  Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(outputs[view])
  inp=f'<schema>{compact(schema)}</schema>\n<text>{c["text"]}</text>'
  tokens=len(sp.encode(inp))+1;assert tokens<=2048,(i,view,tokens)
  out.append(dict(id=f's{i+1:02}_{view}',input=inp,output=compact(outputs[view]),metadata=dict(family_id=f's{i+1:02}',scenario=c['topic'],view=view,split=split,origin='assistant_authored_synthetic',input_tokens_mt5=tokens,tokenizer='google/mt5-small@c83db3e90edd4c0bc69ad9f075c3d19b1ef56c45')))
 return out
def readme(n):return f'''# Изменение состояния и история событий

Опубликовано {n} из 200 пар. План: 50 самостоятельных русских текстов × 4 схемы, а не 200 независимых текстов. Подтверждённые передачи, возвраты, смена ответственных, отмены, откаты, исправления сообщений, роли и доступы, статусы, местонахождение и сроки. Требуется применить последовательность реальных переходов, а не взять последнее упомянутое значение.

`current_state` — итог; `timeline` — начало, переходы с цитатами и итог; `net_changes` — только разница начала и конца (возвраты исключаются); `field_histories` — отдельная история каждого поля, включая неизменившиеся. Только `<schema>…</schema>\n<text>…</text>` на входе; правила и отслеживаемые поля определяет схема. `output` — строка с JSON. `metadata` не подавать модели.

Склеивать batch_*.jsonl построчно. Split по целому источнику: 40 train, 5 validation, 5 test (160/20/20 пар). Все четыре представления одного текста остаются вместе; схемы совпадают между частями, поэтому это не тест неизвестных схем. В этих данных значения строковые, арифметики платежей и остатков нет. Неназванное значение — null. «Нет доступа» и «не назначен» — явные состояния, не null. Отмена предложения сама по себе не восстанавливает предыдущего владельца; требуется явный факт. Ошибочная запись после явного исправления не включается как реальное событие. История упорядочена по событийному времени, а не порядку предложений при наличии временных меток.

Данные синтетические, тексты и разметка написаны ассистентом. Начало, переходы и итог заданы явно; построитель проверяет совпадение отдельно заданного итога с применением переходов, дословность evidence, JSON Schema, split и полный вход ≤2048 по pinned mT5. Независимой человеческой проверки или измеренного прироста качества нет. Источники умеренной длины; это не реальные стенограммы и не корпус полноразмерных совещаний. Часть текстов содержит прямые уточнения, поэтому не все примеры одинаково сложны. Для своего tokenizer длины перепроверить.

Воспроизведение: pip install sentencepiece jsonschema; скачать spiece.model из google/mt5-small на ревизии c83db3e90edd4c0bc69ad9f075c3d19b1ef56c45; задать T5SET_SPIECE_MODEL; запустить scripts/build_batch.py с номером партии 1–20. scripts/cases.py содержит исходную разметку. Токены включают один EOS. Отслеживаемые ключи в схеме задают scope, но начальные/конечные значения в схему не помещаются.
'''
if __name__=='__main__':
 b=int(sys.argv[1]);assert 1<=b<=20
 allrows=[r for i in range(len(CASES)) for r in rows(i)]
 rr=allrows[(b-1)*10:b*10];assert len(rr)==10
 content=''.join(compact(x)+'\n' for x in rr)
 (W/f'batch_{b:03}.jsonl').write_text(content)
 entries=[dict(path=f'{ROOT}/batch_{b:03}.jsonl',mode='100644',type='blob',content=content),dict(path=f'{ROOT}/README.md',mode='100644',type='blob',content=readme(b*10))]
 if b==1:
  entries += [dict(path=f'{ROOT}/scripts/{name}',mode='100644',type='blob',content=(W/name).read_text()) for name in ['cases.py','build_batch.py']]
 if b==20:
  report=dict(pairs=len(allrows),sources=len(CASES),views=4,transitions=sum(len(c['events']) for c in CASES),empty_histories=sum(not c['events'] for c in CASES),min_input_tokens=min(r['metadata']['input_tokens_mt5'] for r in allrows),max_input_tokens=max(r['metadata']['input_tokens_mt5'] for r in allrows),splits={s:sum(r['metadata']['split']==s for r in allrows) for s in ['train','validation','test']},independent_human_review=False)
  (W/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  entries += [dict(path=f'{ROOT}/quality_report.json',mode='100644',type='blob',content=(W/'quality_report.json').read_text()),dict(path=f'{ROOT}/scripts/cases.py',mode='100644',type='blob',content=(W/'cases.py').read_text())]
 print(compact(dict(entries=entries,rows=len(rr),max_tokens=max(r['metadata']['input_tokens_mt5'] for r in rr))))
