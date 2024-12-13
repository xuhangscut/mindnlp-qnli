import pandas as pd
from mindnlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def predict_qnli(model, tokenizer, question, sentence):
    inputs = tokenizer(question, sentence, return_tensors="ms", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    return logits.argmax(axis=1).asnumpy()[0]

if __name__ == '__main__':
    test_file = './QNLI/dev.tsv'
    df = pd.read_csv(test_file, sep='\t', header=0, names=['idx', 'question', 'sentence', 'label'])

    df = df.dropna(subset=['label'])

    label_map = {'entailment': 0, 'not_entailment': 1}
    valid_data = df[df['label'].isin(label_map.keys())]

    questions = valid_data['question'].tolist()
    sentences = valid_data['sentence'].tolist()
    labels = [label_map[label] for label in valid_data['label']]

    tokenizer = AutoTokenizer.from_pretrained("tmnam20/xlm-roberta-large-qnli-1")
    model=AutoModelForSequenceClassification.from_pretrained("tmnam20/xlm-roberta-large-qnli-1", num_labels=2)
    print("model name:"+model.config.model_type)

    predict_true = 0
    for question, sentence, true_label in tqdm(zip(questions, sentences, labels), total=len(questions), desc="Predicting"):
        pred_label = predict_qnli(model, tokenizer, question, sentence)
        if pred_label == true_label:
            predict_true += 1

    accuracy = float(predict_true / len(questions) * 100)
    print(f"测试集总样本数: {len(questions)}")
    print(f"预测正确的数量: {predict_true}")
    print(f"准确率为: {accuracy:.2f}%")

