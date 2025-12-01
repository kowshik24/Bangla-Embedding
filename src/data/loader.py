import json
import os

def load_and_parse_banglarqa(filepath):
    """Parses the nested BanglaRQA JSON file into a flat list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for article in data['data']:
        passage_id = article['passage_id']
        context = article['context']
        for qa in article['qas']:
            example = {
                'passage_id': passage_id,
                'context': context,
                'question_id': qa['question_id'],
                'question_text': qa['question_text'],
                'is_answerable': qa['is_answerable']
            }
            examples.append(example)
    return examples
