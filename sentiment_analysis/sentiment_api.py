from transformers                import AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento               import create_analyzer
from fastapi                     import FastAPI
import pandas   as pd
import numpy    as np
import torch

app = FastAPI(title="Sentiment analysis",
              description="""Analyze the sentiment of a set of comments/tweets""",
              version="0.1.0",)


# Import models
tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
language_detector = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
analyzer_en = create_analyzer(task="sentiment", lang="en")
analyzer_sp = create_analyzer(task="sentiment", lang="es")

def rescale_probs(proba_dict):
    keys, values = list(proba_dict.keys()), list(proba_dict.values())
    pred_key = keys[np.argmax(values)]
    pred_value = 0
    if pred_key == "NEG":
        pred_value = 1-np.max(values)
    elif pred_key == "POS":
        pred_value = np.max(values)
    else:
        neg_value = values[0]
        pos_value = values[-1]

        add_val = pos_value if pos_value > neg_value else -neg_value

        pred_value = 0.5 + (1-np.max(values))*add_val/2
    return pred_value
    
def preprocess_comment_adv(string_, max_word_count=50):
    substrs_to_remove = ["cara emoji", "emoji", "   ", "\n"]
    procs_str = preprocess_tweet(string_)
    for substr in substrs_to_remove:
        procs_str = procs_str.replace(substr, "")
    procs_str = procs_str.replace("Jjaja", "Jajaja")
    
    return " ".join(procs_str.split(" ")[:max_word_count])

@app.get("/")
def home():
    return {"API":"Sentiment analysis of comments/tweets"}


@app.get('/api/status')
def status():
    """
    GET method for API status verification.
    """
    
    message = {
        "status": 200,
        "message": [
            "This API is up and running!"
        ]
    }
    return message

@app.post('/api/get_sentiment')
def get_sentiment(query: dict):
    comments = query["comments"]
    if len(comments) == 0: 
        return {"sentiment": 0.2}
    else:
        preprocessed_comments = [preprocess_comment_adv(comment) for comment in comments]
        languages_detected = []
        for comment in preprocessed_comments:
            inputs = tokenizer(comment, return_tensors="pt")
            with torch.no_grad():
                logits = language_detector(**inputs).logits
            predicted_class_id = logits.argmax().item()
            languages_detected.append(language_detector.config.id2label[predicted_class_id])

        sentiment_probas = []
        for ix in range(len(preprocessed_comments)):
            if languages_detected[ix] == "en":
                estimation = analyzer_en.predict(preprocessed_comments[ix]).probas
            elif languages_detected[ix] == "es" or languages_detected[ix] == "pt":
                estimation = analyzer_sp.predict(preprocessed_comments[ix]).probas
            else:
                estimation = {'NEG': 0, 'NEU': 1, 'POS': 0}
            sentiment_probas.append(estimation)

        sentiment = [rescale_probs(probs) for probs in sentiment_probas]
        df = pd.DataFrame({"Comment": comments, "Language": languages_detected, "Sentiment": sentiment})

        return {"sentiment": df["Sentiment"].mean()}