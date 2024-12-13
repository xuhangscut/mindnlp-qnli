import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm


def predict_qnli(model, tokenizer, device, question, sentence):
    inputs = tokenizer(question, sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    return logits.argmax(dim=1).item()

if __name__ == '__main__':
    test_file = './QNLI/dev.tsv'
    df = pd.read_csv(test_file, sep='\t', header=0, names=['idx', 'question', 'sentence', 'label'])

    df = df.dropna(subset=['label'])

    label_map = {'entailment': 0, 'not_entailment': 1}
    valid_data = df[df['label'].isin(label_map.keys())]

    questions = valid_data['question'].tolist()
    sentences = valid_data['sentence'].tolist()
    labels = [label_map[label] for label in valid_data['label']]

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = AutoModelForSequenceClassification.from_pretrained("ModelTC/bart-base-qnli", num_labels=2)
    print("model name:"+model.config.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    predict_true = 0
    for question, sentence, true_label in tqdm(zip(questions, sentences, labels), total=len(questions), desc="Predicting"):
        pred_label = predict_qnli(model, tokenizer, device, question, sentence)
        if pred_label == true_label:
            predict_true += 1


    accuracy = float(predict_true / len(questions) * 100)
    print(f"测试集总样本数: {len(questions)}")
    print(f"预测正确的数量: {predict_true}")
    print(f"准确率为: {accuracy:.2f}%")
