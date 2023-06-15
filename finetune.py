from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

# raw data comes as a dict with these keys and values:
# - question: str
# - human_answers: list(str)
# - chatgpt_answers: list(str)
# - index: ???
# - source: str
raw_data = load_dataset("json", data_files="HC3/all.jsonl")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# create dataset of 
def format_dataset(raw_data):
    answers = []
    for example in raw_data["train"]:
        for human_answer in example["human_answers"]:
            answers.append({"label": 0, "text": human_answer})
        for gpt_answer in example["chatgpt_answers"]:
            answers.append({"label": 1, "text": gpt_answer})
    return Dataset.from_list(answers)

answers = format_dataset(raw_data)

tokenized_answers = answers.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

answers_split = tokenized_answers.train_test_split()

training_args = TrainingArguments(
    output_dir='results',
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=answers_split['train'],
    eval_dataset=answers_split['test'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

result = trainer.train()
print(result)