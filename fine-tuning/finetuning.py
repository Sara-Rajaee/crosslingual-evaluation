import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--MODEL")
parser.add_argument("--TASK")
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_crosslingual_eval", action='store_true')
parser.add_argument("--SAVING_PATH")
parser.add_argument("--SAVED_MODEL_PATH", default="")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--STRIDE", default=128, type=int)
parser.add_argument("--BATCH", default=16, type=int)
parser.add_argument("--LR", default=9e-6, type=float)
parser.add_argument("--WARMUP_RATIO", default=0.01, type=float)
parser.add_argument("--WEIGHT_DECAY", default=1e-6, type=float)
parser.add_argument("--EPOCHS", default=3, type=int)
parser.add_argument("--SEED", default=42, type=int)
parser.add_argument("--SHUFFLE", action='store_true')
parser.add_argument("--per_label_evaluation", action='store_true')

args = parser.parse_args()

SAVING_PATH = "/ivi/ilps/personal/srajaee/bias/"

MODEL_TO_CASING = { "XLM-r": "xlm-roberta-base",
                    "mBERT": "bert-base-multilingual-cased",
                    "INFoXLM": "microsoft/infoxlm-base"}

from transformers import AutoConfig, set_seed, AutoModelForSequenceClassification,BertConfig, \
    AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, \
    Trainer
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))
from utils.qa_utils import compute_metrics, preprocess_training_examples, preprocess_validation_examples, \
    Squad, preprocess_overlap_examples, compute_metrics_persample
from utils.classification_utils import data_preprocessing, do_shuffle
import evaluate
from datasets import load_dataset
import numpy as np


set_seed(args.SEED)

def loading():
    if args.do_crosslingual_eval and not args.do_train:
        casing = args.SAVED_MODEL_PATH
    else:
        casing = MODEL_TO_CASING[args.MODEL]


    config = AutoConfig.from_pretrained(casing)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_CASING[args.MODEL], add_special_tokens=True)
    TASK = args.TASK
    
    data_collator = DefaultDataCollator()
    if TASK=='xnli':
        config.num_labels = 3
        model = AutoModelForSequenceClassification.from_pretrained(casing,
            config=config)
        dataset = load_dataset(TASK, 'en')
    elif TASK=='paws-x':
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(casing,
            config=config)
        dataset = load_dataset(TASK, 'en')
    elif TASK=='squad':
        model = AutoModelForQuestionAnswering.from_pretrained(casing)
        dataset = load_dataset(TASK)
    

    return tokenizer, model, dataset,data_collator, TASK


def training():

    tokenizer, model, dataset,data_collator, TASK = loading()

    training_args = TrainingArguments(output_dir=args.SAVING_PATH+args.MODEL+"/"+TASK+"/"+str(args.SEED)+"/",
                                    do_train=True,
                                    do_eval =True,
                                    evaluation_strategy="epoch",
                                    logging_steps=1000,
                                    save_strategy="epoch",
                                    seed=args.SEED,
                                    optim="adamw_torch",
                                    learning_rate=args.LR,
                                    weight_decay=args.WEIGHT_DECAY,
                                    num_train_epochs=args.EPOCHS,
                                    per_device_train_batch_size=args.BATCH,
                                    remove_unused_columns=True,
                                    warmup_ratio=args.WARMUP_RATIO,
                                    )

    ## Preprocessing data and setting trainer arguments

    if args.SHUFFLE:
            dataset = do_shuffle(dataset, TASK)

    if TASK == 'xnli':

            train_loader = data_preprocessing(dataset["train"], tokenizer, TASK, args.MAX_LENGTH)
            valid_loader = data_preprocessing(dataset['validation'],tokenizer, TASK, args.MAX_LENGTH)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_loader,
                eval_dataset=valid_loader,
            )
            languages = ['en', 'es', 'de', 'ar', 'ur', 'ru', 'bg', 'el', 'fr', 'hi', 'sw', 'th', 'tr', 'vi', 'zh']
            metric = evaluate.load("accuracy")

    elif TASK == 'paws-x':

            train_loader = data_preprocessing(dataset["train"], tokenizer, TASK, args.MAX_LENGTH)
            valid_loader = data_preprocessing(dataset['validation'], tokenizer, TASK, args.MAX_LENGTH)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_loader,
                eval_dataset=valid_loader,
            )
            languages = ['en', 'es', 'de', 'fr', 'zh', 'ja', 'ko']
            metric = evaluate.load("accuracy")

    elif TASK == 'squad':

            train_loader = dataset["train"].map(
            preprocess_training_examples,
            fn_kwargs={"tokenizer": tokenizer, 'max_length': args.MAX_LENGTH, 'stride': args.STRIDE},
            batched=True,
            remove_columns=dataset["train"].column_names)

            valid_loader = dataset["validation"].map(
                preprocess_validation_examples,
                fn_kwargs={"tokenizer": tokenizer, 'max_length': args.MAX_LENGTH, 'stride': args.STRIDE},
                batched=True,
                remove_columns=dataset["validation"].column_names,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_loader,
                eval_dataset=valid_loader,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            langueges = ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']
            metric = evaluate.load("squad")
            


    if args.do_train:
        trainer.train()

    if args.do_crosslingual_eval:
        print("***************  "+TASK + " Evaluation on " + args.MODEL + "  ***************", file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
        if TASK=='xnli':
            
            for l1 in languages:
                valid_dataset = load_dataset(TASK, l1)['test']

                for l2 in languages: 
                    print("premise language is :", l1, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    print("Hypothesis language is :", l2, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    valid_dataset = valid_dataset.remove_columns("hypothesis")
                    valid_dataset_2 = load_dataset(TASK, l2)['test']
                    valid_dataset = valid_dataset.add_column("hypothesis", valid_dataset_2['hypothesis'])
                    valid_loader = data_preprocessing(valid_dataset, tokenizer, TASK, args.MAX_LENGTH)

                    trainer = Trainer(
                            model=model,
                            args=training_args,
                            eval_dataset=valid_loader,
                        )

                
                    results = trainer.predict(valid_loader)
                    pre = np.argmax(results[0], axis=1)
                    print(metric.compute(predictions=pre, references=results.label_ids), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    
                    if args.per_label_evaluation:
                        ent = [pre[i] for i in range(len(valid_dataset['label'])) if valid_dataset['label'][i] == 0 or valid_dataset['label'][i] == 1]
                        ent_label = [i for i in valid_dataset['label'] if i ==0 or i ==1]
                        nent = [pre[i] for i in range(len(valid_dataset['label'])) if valid_dataset['label'][i] == 2 ]
                        nent_label = [i for i in valid_dataset['label'] if i ==2]
                        print("Entailment performance : ", metric.compute(predictions=ent, references=ent_label), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                        print("Not Entailment performance : ", metric.compute(predictions=nent, references=nent_label), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))

        elif TASK=='paws-x':
            for l1 in languages:
                for l2 in languages: 
                    valid_dataset = load_dataset(TASK, l1)['test']
                    valid_dataset = valid_dataset.remove_columns("sentence2")    
                    valid_dataset_2 = load_dataset(TASK, l2)['test']
                    sec_sentence = []
                    tmp = []
                    ## discarding a few examples in PAWS_X dataset which are not aligned across languages
                    for i in range(len(valid_dataset)):
                        if valid_dataset['label'][i]!= valid_dataset_2['label'][i]:
                            continue
                        else:
                            tmp.append(i)
        
                    valid_dataset = valid_dataset.select(tmp)
                    valid_dataset_2 = valid_dataset_2.select(tmp)
                    valid_dataset = valid_dataset.add_column("sentence2", valid_dataset_2['sentence2'])
                    valid_loader = data_preprocessing(valid_dataset, tokenizer, TASK, args.MAX_LENGTH)
                    trainer = Trainer(
                            model=model,
                            args=training_args,
                            eval_dataset=valid_loader,
                        )
                    results = trainer.predict(valid_loader)
                    pre = np.argmax(results[0], axis=1)
                    print("Sentence 1 Language is: ", l1, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    print("Sentence 2 Language is: ", l2, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    print(metric.compute(predictions=pre, references=results.label_ids), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))

                    if args.per_label_evaluation:
                        par = [pre[i] for i in range(len(valid_dataset['label'])) if valid_dataset['label'][i] == 0 ]
                        par_label = [i for i in valid_dataset['label'] if i ==0 ]
                        npar = [pre[i] for i in range(len(valid_dataset['label'])) if valid_dataset['label'][i] == 1 ]
                        npar_label = [i for i in valid_dataset['label'] if i ==1]
                        print("Paraphrase performance : ", metric.compute(predictions=par, references=par_label), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                        print("Non Paraphrase performance : ", metric.compute(predictions=npar, references=npar_label), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))


        elif TASK=='squad':
            persample_trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=valid_loader,
                tokenizer=tokenizer,
            )
            for l_context in langueges:
                dataset = 'xquad'
                data_lang = 'xquad.'+l_context
                data = load_dataset(dataset, data_lang)
                for l_question in langueges:
                    print('Context Language is :', l_context, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    print('Question Language is :', l_question, file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))
                    data_lang = 'xquad.' + l_question
                    data_context = load_dataset(dataset, data_lang)
                    data['validation'] = data['validation'].remove_columns(["question"])
                    data['validation'] = data['validation'].add_column("question", data_context['validation']['question'])
                    validation_dataset = data["validation"].map(
                        preprocess_validation_examples,
                        fn_kwargs={"tokenizer": tokenizer, 'max_length': args.MAX_LENGTH, 'stride': args.STRIDE},
                        batched=True,
                        remove_columns=data["validation"].column_names,
                    )
                    predictions, _, _ = trainer.predict(validation_dataset)
                    start_logits, end_logits = predictions
                    print(compute_metrics(start_logits, end_logits, validation_dataset, data["validation"], metric), file=open(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_results.o", 'a+'))

                    if args.per_label_evaluation:

                        validation_dataset = data_context["validation"].map(
                        preprocess_overlap_examples,
                        fn_kwargs={"tokenizer": tokenizer, 'max_length': args.MAX_LENGTH, 'stride': args.STRIDE},
                        batched=True,
                        remove_columns=data_context["validation"].column_names,
                        )
                        print(validation_dataset)
                        predictions, _, _ = persample_trainer.predict(validation_dataset)
                        start_logits, end_logits = predictions
                        y, overlap = compute_metrics_persample(start_logits, end_logits, validation_dataset, data_context["validation"])
                        np.save(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_exact_match_"+l_context+"_"+l_question+".npy", np.asarray(y['exact_match']))
                        np.save(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_f1_"+l_context+"_"+l_question+".npy", np.asarray(y['f1']))
                        np.save(args.SAVING_PATH+args.MODEL+"_"+TASK+"_"+str(args.SEED)+"_distance_"+l_context+"_"+l_question+".npy", np.asarray(overlap))
                        
if __name__ == "__main__":
    training()