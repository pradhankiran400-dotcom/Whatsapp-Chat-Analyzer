import re
import pandas as pd

def prepocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[aApP][mM]\s-\s'

    message = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': message, 'message_date': dates})
    # Preprocess message_date to handle non-breaking space
    df['message_date'] = df['message_date'].str.replace('\u202f', ' ')
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # separate users name
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('(.*?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])

        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['users'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day_name'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df