from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import emoji
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user == 'Overall':
        #fetch number of message
        num_messages = df.shape[0]
        #fetch number of words
        words = []
        for message in df['message']:
            words.extend(message.split())

        num_of_media = df[df['message']=='<Media omitted>\n'].shape[0]

        links = []
        for message in df['message']:
            links.extend(extract.find_urls(message))

        return num_messages, len(words), num_of_media,len(links)


    else:
        new_df = df[df['users'] == selected_user]
        words = []
        for message in new_df['message']:
            words.extend(message.split())

        links = []
        for message in new_df['message']:
            links.extend(extract.find_urls(message))

        num_of_media = new_df[new_df['message']=='<Media omitted>\n'].shape[0]
        return df[df['users'] == selected_user].shape[0],len(words),num_of_media,len(links)

def most_busy_user(df):
    X = df['users'].value_counts().head()
    df = round((df['users'].value_counts() / df.shape[0])*100,2).reset_index().rename(columns={'index':'name','users':'percent'})
    return X,df

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.update(['Media', 'omitted', 'null', 'message', 'deleted'])

wc = WordCloud(width=500, height=500, min_font_size=10,
               background_color='white',
               stopwords=stopwords)

def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]


    df = df[df['message'] != '<Media omitted>\n']

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    from collections import Counter
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " " + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]

    return df['month'].value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

def train_user_identifier(df, model_type='nb'):
    # Filter out group notifications and media
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp = temp[temp['message'].str.strip() != '']

    # Need at least 2 users with enough messages
    user_counts = temp['users'].value_counts()
    valid_users = user_counts[user_counts >= 20].index.tolist()
    temp = temp[temp['users'].isin(valid_users)]

    if len(valid_users) < 2:
        return None, None, "Not enough data. Each user needs at least 20 messages."

    X = temp['message']
    y = temp['users']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'nb':
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('model', MultinomialNB())
        ])
    else:  # SVM
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('model', SVC(kernel='linear', probability=True))
        ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, accuracy, report


def predict_user(model, message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, np.round(probabilities * 100, 2)))
    return prediction, prob_dict


