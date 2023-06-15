from datasets import load_dataset, Dataset
import nltk
nltk.download('punkt')
from nltk import sent_tokenize # for spliting English sentences
import argparse

def prepare(data):
    answers = []
    for example in data["train"]:
        for human_answer in example["human_answers"]:
            answers.append({"label": 0,
                            "text": human_answer,
                            "question": example["question"],
                            "source": example["source"]})
        for gpt_answer in example["chatgpt_answers"]:
            answers.append({"label": 1, 
                            "text": gpt_answer,
                            "question": example["question"],
                            "source": example["source"]})
    return Dataset.from_list(answers)

def split(data):
    sentences = []
    for answer in data:
        for sentence in sent_tokenize(answer["text"]):
            sentences.append({"label": answer["label"],
                              "text": sentence,
                              "question": answer["question"],
                              "source": answer["source"]})
    return Dataset.from_list(sentences)

# from https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/HC3/demo_indicating_words.ipynb
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")
def replace_ni(string):
    return string.replace('您','你')
def filtering(text, indicating_words, language, verbose=False):
    '''removing sentence(s) that includes indicating words'''
    assert isinstance(text, str)
    assert isinstance(indicating_words, list)
    if language == 'en':
        sents = sent_tokenize(text)
    elif language == 'zh':
        sents = cut_sent(text)
    else:
        raise NotImplementedError
  
    filtered_sents = []
    for s in sents:
        if language == 'zh':
            # replace"您"to"你" for Chinese corpus
            s = replace_ni(s)
        has = False
        for k in indicating_words:
            if k in s:
                has = True
                break
        if not has:
            filtered_sents.append(s)
            
    filtered_sents = ' '.join(filtered_sents)
    
    if verbose:
        print(f'Original answers: {text} \nFiltered answers: {filtered_sents}\n')

    return filtered_sents

def filter(data):
    # TODO
    return data

def add_question(data):
    # TODO
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("-f", "--filter", action="store_true")
    parser.add_argument("-l", "--level", type=str)
    parser.add_argument("-qa", "--qa", action="store_true")
    args = parser.parse_args()

    data = load_dataset(args.data)
    data = prepare(data)

    if args.filter:
        data = filter(data)
    
    if args.level == "full":
        pass
    elif args.level == "sent":
        data = split(data)
    elif args.level == "mix":
        # TODO implement
        pass

    if args.qa:
        data = add_question(data)

    # TODO output dataset

if __name__ == "__main__":
    main()