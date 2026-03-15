import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import pandas as pd

from helper import most_common_words

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 400px;
            background-color: #075E54;
             
        }
        
    </style>
""", unsafe_allow_html=True)

st.sidebar.title('WhatsApp Chat Analyzer')
uploaded_file = st.sidebar.file_uploader('Choose a file')

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.prepocess(data)

    # st.dataframe(df)

    #fetch unique users
    user_list = df['users'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,'Overall')

    selected_user = st.sidebar.selectbox('Select a user', user_list)

    # st.sidebar.selectbox('Show analysis wrt',user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages,words,num_of_media,num_of_Links= helper.fetch_stats(selected_user,df)

        col1,col2,col3,col4 = st.columns(4)

        with col1:
             st.header('Total Message')
             st.title(num_messages)

        with col2:
            st.header('Total Words')
            st.title(words)

        with col3:
            st.header('Total  Files📝')
            st.title(num_of_media)

        with col4:
            st.header('Total Links')
            st.title(num_of_Links)

    # finding the busiet user in the group
    if selected_user == 'Overall':
        st.title('Most Busy User')
        X, new_df_percent = helper.most_busy_user(df)

        # Color picker in sidebar
        bar_color = st.color_picker('Pick Bar Color', '#25D366')

        fig, ax = plt.subplots(figsize=(10, 5))
        col1, col2 = st.columns(2)

        with col1:
            ax.bar(X.index, X.values, color=bar_color)
            st.pyplot(fig)

        with col2:
            st.dataframe(new_df_percent)

    df_wc = helper.create_wordcloud(selected_user,df)
    fig,ax = plt.subplots(figsize=(10, 5))
    ax.imshow(df_wc)
    st.pyplot(fig)

    #most common words
    st.title("Most Common Words")
    most_common_df =  helper.most_common_words(selected_user,df)
    fig,ax = plt.subplots()
    ax.barh(most_common_df[0],most_common_df[1])
    st.pyplot(fig)

    #Emoji Analysis
    col1, col2 = st.columns(2)
    emoji_df = helper.emoji_helper(selected_user,df)
    with col1:
        st.dataframe(emoji_df)

    #timeline
    st.title("Monthly Timeline")
    timeline = helper.monthly_timeline(selected_user,df)
    fig,ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    #activiy map week
    st.title('Activity Map')
    col1, col2 = st.columns(2)

    with col1:
        st.header('Most busy day')
        busy_day = helper.week_activity_map(selected_user,df)
        fig,ax = plt.subplots()
        ax.bar(busy_day.index,busy_day.values)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.header('Most busy month')
        busy_month = helper.month_activity_map(selected_user,df)
        fig,ax = plt.subplots()
        ax.bar(busy_month.index,busy_month.values)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ---- User Identification Section ----
st.title(" Who Sent This Message? (ML Prediction)")

model_choice = st.radio("Choose Model", ["Naive Bayes (Fast)", "SVM (Accurate)"])
model_type = 'nb' if 'Naive Bayes' in model_choice else 'svm'

if st.button("Train Model"):
    with st.spinner("Training..."):
        model, accuracy, report = helper.train_user_identifier(df, model_type)

    if model is None:
        st.error(report)  # error message
    else:
        st.session_state['model'] = model
        st.success(f"Model trained! Accuracy: {round(accuracy * 100, 2)}%")

        # Show per-user precision/recall
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

# Predict on custom message
if 'model' in st.session_state:
    st.subheader(" Test the Model")
    user_input = st.text_input("Type any message:")

    if user_input:
        predicted_user, prob_dict = helper.predict_user(
            st.session_state['model'], user_input
        )
        st.success(f"**Predicted Sender: {predicted_user}**")

        # Show probability bar chart
        prob_df = pd.DataFrame(
            list(prob_dict.items()), columns=['User', 'Confidence (%)']
        ).sort_values('Confidence (%)', ascending=False)

        fig, ax = plt.subplots()
        ax.barh(prob_df['User'], prob_df['Confidence (%)'], color='#25D366')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence per User")
        st.pyplot(fig)
