import random

def do_shuffle(dataset, task):

    size = len(dataset['train'])
    if task == 'xnli':

        premise = dataset['train']['premise']
        hypothesis = dataset['train']['hypothesis']
        dataset['train'] = dataset['train'].remove_columns("hypothesis")
        dataset['train'] = dataset['train'].remove_columns("premise")

        premise = [' '.join(random.sample(p.split(),len(p.split()))) for p in premise ]
        hypothesis = [' '.join(random.sample(h.split(), len(h.split()))) for h in hypothesis ] 

        dataset['train'] = dataset['train'].add_column("hypothesis", hypothesis)
        dataset['train'] = dataset['train'].add_column("premise", premise)

    elif task == 'paws-x':

        sentence1 = dataset['train']['sentence1']
        sentence2 = dataset['train']['sentence2']
        
        dataset['train'] = dataset['train'].remove_columns("sentence2")
        dataset['train'] = dataset['train'].remove_columns("sentence1")

        sentence1 = [' '.join(random.sample(s.split(),len(s.split()))) for s in sentence1 ]
        sentence2 = [' '.join(random.sample(s.split(), len(s.split()))) for s in sentence2 ]

        dataset['train'] = dataset['train'].add_column("sentence2", sentence2)
        dataset['train'] = dataset['train'].add_column("sentence1", sentence1)

    elif task == 'squad':
        
        questions = dataset['train']['question']
        dataset['train'] = dataset['train'].remove_columns("question")
        questions = [' '.join(random.sample(q.split(), len(q.split()))) for q in questions]
        dataset['train'] = dataset['train'].add_column("question", questions)

    else:
        raise Exception("The task is not supported!")
   
    return dataset

def encode(examples, tokenizer, task, max_length):
    if task == 'xnli':

        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length',
                        max_length=max_length, return_attention_mask=True)

    elif task == 'paws-x':

        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length',
                        max_length=max_length, return_attention_mask=True)
    else:

        raise Exception("The task is not supported!")

def data_preprocessing(input, tokenizer, task, max_length):

    inputs = input.map(encode,fn_kwargs={'tokenizer': tokenizer, 'task': task, "max_length": max_length}, batched=True)
    inputs = inputs.rename_column("label", "labels")
    inputs.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return inputs