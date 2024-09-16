import json
import math
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, EarlyStoppingCallback)
from arg_parser import parse_args
from preprocess import lower_case_func


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN,
                     return_tensors='pt')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def load_model_and_tokenizer(model_path, num_labels, max_len, device="cuda"):
    # LOAD MODEL AND TOKENIZER
    load_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    load_model = load_model.to(device)
    load_tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_len)
    return load_tokenizer, load_model


if __name__ == "__main__":
    # Get the environment variables
    args = parse_args()

    df_train = pd.read_csv("data/stance_train.csv")
    df_val = pd.read_csv("data/stance_val.csv")
    df_test = pd.read_csv("data/stance_test.csv")

    MAX_LEN = args.max_len

    # Check how many labels are there in the dataset
    unique_labels = df_train.labels.unique().tolist()

    # Map each label into its id representation and vice versa
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    print(ids_to_labels)

    df_train['labels'] = df_train.labels.apply(lambda x: labels_to_ids[x]).tolist()
    df_val['labels'] = df_val.labels.apply(lambda x: labels_to_ids[x]).tolist()
    df_test['labels'] = df_test.labels.apply(lambda x: labels_to_ids[x]).tolist()

    print("Train data shape: ", df_train.shape)
    print("Val. data shape: ", df_val.shape)
    print("Test data shape: ", df_test.shape)

    # Token Coverage
    MODEL_NAME = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LEN)

    total_token_count = 0
    unk_token_count = 0

    for index, dp in df_train.iterrows():
        tokenized_text = tokenizer.tokenize(str(dp['text']))
        unk_token_count += len([i for i in tokenized_text if i[0:2] == "##"])
        total_token_count += len(tokenized_text)

    print(f"Percentage of tokens unknown: {(100.0 * unk_token_count / total_token_count)}")

    if "uncased" in MODEL_NAME:
        df_train['text'] = df_train.text.apply(lower_case_func).tolist()
        df_val['text'] = df_val.text.apply(lower_case_func).tolist()
        df_test['text'] = df_test.text.apply(lower_case_func).tolist()

    dataset_train = Dataset.from_pandas(df_train[["text", "labels"]], split="train")
    dataset_val = Dataset.from_pandas(df_val[["text", "labels"]], split="val")
    dataset_test = Dataset.from_pandas(df_test[["text", "labels"]], split="test")

    train_dataset = dataset_train.map(preprocess_function, batched=True, num_proc=2, remove_columns=["text"],
                                      fn_kwargs={"tokenizer": tokenizer})
    val_dataset = dataset_val.map(preprocess_function, batched=True, remove_columns=["text"],
                                  fn_kwargs={"tokenizer": tokenizer})
    test_dataset = dataset_test.map(preprocess_function, batched=True, num_proc=2, remove_columns=["text"],
                                    fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=len(unique_labels))

    EPOCH = args.epoch
    warmup_steps = math.ceil(len(train_dataset) * EPOCH * 0.1)
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    WD = args.weight_decay

    training_args = TrainingArguments(
        num_train_epochs=EPOCH,

        # Optimizer Hyperparameters
        optim="adamw_torch",
        learning_rate=LR,
        weight_decay=WD,
        warmup_steps=warmup_steps,

        # Logging Hyperparameters
        run_name="stance-detection",
        output_dir=args.checkpoint,
        overwrite_output_dir=True,
        logging_steps=250,
        evaluation_strategy="steps",
        save_strategy="steps",

        # Weight and Biases
        report_to="none",

        # General Hyperparameters
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=1,
        do_train=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train(resume_from_checkpoint=False)
    print(f"Results for {MODEL_NAME}")
    print("*"*30)
    results = trainer.evaluate(eval_dataset=test_dataset)
    for key, value in results.items():
        print(f"{key} = {value}")

    # SAVE MODEL
    prefix = MODEL_NAME.replace("dbmdz/", "")
    save_path = f"{args.save_dir}/{prefix}"

    trainer.save_model(f'{save_path}')

    # Save Parameters
    with open(f"{save_path}/parameters.txt", "w+", encoding="utf-8") as f:
        f.write(f"MODEL NAME: {MODEL_NAME}\n")
        f.write(f"MAX LEN: {MAX_LEN}\n")
        f.write(f"EPOCH: {EPOCH}\n")
        f.write(f"BATCH SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"WD: {WD}\n")

    with open(f"{save_path}/id2label.json", "w+", encoding="utf-8") as fp:
        json.dump(ids_to_labels, fp, indent=4)

    with open(f"{save_path}/label2id.json", "w+", encoding="utf-8") as fp:
        json.dump(labels_to_ids, fp, indent=4)

    tokenizer, loaded_model = load_model_and_tokenizer(save_path, len(labels_to_ids), MAX_LEN)

    if args.hf_repo_name:
        tokenizer.push_to_hub(args.repo_name)
        loaded_model.push_to_hub(args.repo_name)
