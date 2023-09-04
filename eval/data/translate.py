import json
from googletrans import Translator
translator = Translator()

def translate_google(value):
    response = None
    while response is None:
        try:
            response = translator.translate(value, dest='vi', src='en')
        except Exception as e:
            print(e)
    print(response.text)
    return response.text
 
lines = []
with open('question.jsonl') as json_file:
    json_list = list(json_file)

with open('question-vi.jsonl', 'w', encoding="utf-8") as outfile:
    for json_str in json_list:
        item = json.loads(json_str)
        item['text'] = translate_google(item['text'])
        item['lang'] = 'vi'
        #json.dump(item, outfile)
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write('\n')
