import argparse, json, re, hashlib, statistics, os
from pathlib import Path
import sentencepiece as spm
from jsonschema import Draft202012Validator
from cases import CASES

W=Path(__file__).parent
ROOT='datasets/noisy_chunks_ru_100'
sp=spm.SentencePieceProcessor(model_file=os.environ.get('T5SET_SPIECE_MODEL',str(W.parents[1]/'meetings_hard_context_work/spiece.model')))
def compact(o):return json.dumps(o,ensure_ascii=False,separators=(',',':'))
def schema(scope):
    return {'description':scope+' Извлеки только текущие поручения по видимому фрагменту после видимых уточнений. Не достраивай отсутствующие реплики. Неназванные владельцы: []; срок: null. committed — явное принятие; requested — только просьба; conditional — принято с условием. Идеи, снятое и завершённое исключить.',
    'type':'array','items':{'type':'object','properties':{'action':{'type':'string'},'owners':{'type':'array','items':{'type':'string'}},'deadline':{'type':['string','null']},'status':{'type':'string','enum':['committed','requested','conditional']},'condition':{'type':['string','null']}},'required':['action','owners','deadline','status','condition'],'additionalProperties':False}}
def records(i):
    c=CASES[i];family=f'n{i+1:02}'
    full=c['before']+'\n'+c['background']+'\n'+c['after']
    no_scope='Все поручения из предоставленного текста.'
    foreign='[Документ PREV: другой разговор]\n'+c['neighbor']+'\n[Конец PREV]\n[Документ MAIN]\n'+full+'\n[Конец MAIN]\n[Документ NEXT: другой разговор]\n'+c['neighbor']+'\n[Конец NEXT]'
    variants=[('clean',full,c['final'],no_scope),
        ('marked_neighbors',foreign,c['final'],'Только поручения документа MAIN; PREV и NEXT исключить.'),
        ('left_cut',c['left'],c['partial'],no_scope),
        ('right_cut',c['before']+'\n'+c['background']+'\n[фрагмент обрывается]',c['old'],no_scope),
        ('punctuation_noise',re.sub(r'[.,;!?]|(?<!\d):(?!\d)','',full).lower(),c['final'],no_scope)]
    rr=[]
    for kind,text,out,scope in variants:
        sch=schema(scope);Draft202012Validator.check_schema(sch);Draft202012Validator(sch).validate(out)
        inp=f'<schema>{compact(sch)}</schema>\n<text>{text}</text>'
        tokens=len(sp.encode(inp))+1
        assert tokens<=2048,(family,kind,tokens)
        for task in out:
            assert task['deadline'] is None or task['deadline'].lower() in text.lower()
            assert all(o.lower() in text.lower() for o in task['owners'])
            assert (task['condition'] is not None)==(task['status']=='conditional')
        rr.append(dict(id=family+'_'+kind,input=inp,output=compact(out),metadata=dict(family_id=family,scenario=c['topic'],variant=kind,split='train' if i<16 else ('validation' if i<18 else 'test'),origin='assistant_authored_synthetic',input_tokens_mt5=tokens,tokenizer='google/mt5-small@c83db3e90edd4c0bc69ad9f075c3d19b1ef56c45')))
    return rr

def readme(n):
    return f'''# Шумные границы и неполные фрагменты

Опубликовано {n} из 100 плановых пар, порциями по 10.
Формат: `<schema>{{JSON Schema}}</schema>\n<text>{{текст}}</text>` → сериализованный JSON в строке `output`.

Один авторский сценарий представлен пятью вариантами: чистый диалог, явно размеченные соседние документы, обрезанное начало, обрезанный конец, потеря пунктуации и регистра. Разметка пересчитывается для каждого видимого фрагмента. Вариант без конца не знает об отмене/смене срока за границей входа; вариант без начала не получает владельца из отсутствующей реплики.

Соседние документы исключаются только при явной границе и указании MAIN в схеме. Без этих сигналов нельзя обучать модель угадывать, какой текст «главный». Пунктуационный шум — упрощённая имитация, не реальные ошибки ASR/OCR; он не заменяет обучение на таких ошибках.

Файлы `batch_*.jsonl` содержат непересекающиеся порции. Склеивайте их построчно, передавайте модели только `input`. Все пять вариантов одной семьи имеют один `metadata.split`: первые 16 семей train, следующие 2 validation, последние 2 test. Не перемешивайте варианты семьи между splits. Это 20 самостоятельных сценариев и 100 вариантов, не 100 независимых встреч.

Источники и ответы сочинены ассистентом на русском. Проверки: JSON Schema, присутствие владельцев и сроков, длина полного входа по указанному mT5 SentencePiece с EOS. Это не независимо проверенный human gold. Лимит 2048 проверен для mT5, для своего checkpoint перепроверьте. Длинные полноформатные стенограммы здесь не моделируются. Улучшение вашей модели пока не измерено.
'''

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument('batch',type=int);a=p.parse_args()
    start=(a.batch-1)*2;assert len(CASES)>=start+2
    rows=records(start)+records(start+1)
    content=''.join(compact(r)+'\n' for r in rows)
    entries=[dict(path=f'{ROOT}/batch_{a.batch:03}.jsonl',mode='100644',type='blob',content=content),dict(path=f'{ROOT}/README.md',mode='100644',type='blob',content=readme(a.batch*10))]
    Path(W/f'batch_{a.batch:03}.jsonl').write_text(content)
    print(compact({'entries':entries,'rows':len(rows),'max_input_tokens_mt5':max(r['metadata']['input_tokens_mt5'] for r in rows)}))
