import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer

train_path = "data/train.csv"
test_path = "data/test.csv"
categories_path = "data/categories.txt"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df = train_df[["text"]].dropna()
test_df = test_df[["text"]].dropna()

with open(categories_path, "r", encoding="utf-8") as f:
    categories = [line.strip() for line in f.readlines()]
#Авторазметка
def keyword_label(text):
    text = str(text).lower()
    if re.search(r"платье|куртк|рубашк|джинс|юбк|пуховик", text):
        return "одежда"
    if re.search(r"ботинк|сапог|кроссовк|обувь|туфл", text):
        return "обувь"
    if re.search(r"чайник|пылесос|микроволн|холодильник|блендер", text):
        return "бытовая техника"
    if re.search(r"телефон|смартфон|ноутбук|планшет|наушник", text):
        return "электроника"
    if re.search(r"тарелк|стакан|ложк|вилк|кастрюл|сковород", text):
        return "посуда"
    if re.search(r"штор|полотенц|простын|подушк|одеял", text):
        return "текстиль"
    if re.search(r"ребенк|детск|игрушк|малыш|кукл", text):
        return "товары для детей"
    if re.search(r"серьг|кольц|браслет|часы|аксессуар", text):
        return "украшения и аксессуары"
    return "нет товара"

train_df["label"] = train_df["text"].apply(keyword_label)
#эмбединг
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X = model_emb.encode(train_df["text"].tolist(), batch_size=64, show_progress_bar=True)
y = train_df["label"]

#разделяю на обучающую выборку и валидационную
class_counts = y.value_counts()
if class_counts.min() < 2:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )