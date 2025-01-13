import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

@st.cache_resource
def load_tokenizer():
    """Load and cache the BERT tokenizer."""
    return BertTokenizer.from_pretrained("./bert_binary_classification_model")

@st.cache_resource
def load_classification_model():
    """Load and cache the BERT sequence classification model."""
    return BertForSequenceClassification.from_pretrained("./bert_binary_classification_model")

@st.cache_resource
def load_chat_llm():
    """Load and cache the ChatGroq language model."""
    return ChatGroq(
        groq_api_key="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",
        model_name="llama-3.1-8b-instant"
    )

def generate_app_description(track_name, genre, currency, price, size_bytes, supported_devices,
                             app_desc, content_rating, language_count, rating_total, ipad_support_count):
    """
    Create a formatted application description based on provided metadata.
    """
    description = (
        f'آپ "{track_name}" یک برنامه {genre} است که به ارز {currency} با قیمت {price:.2f} عرضه می‌شود. '
        f'این برنامه دارای حجمی تقریبی برابر با {size_bytes / (1024**2):.1f} مگابایت بوده و از {supported_devices} دستگاه پشتیبانی می‌کند. '
        f'توضیحات برنامه: "{app_desc}" '
        f'مناسب برای کاربران {content_rating}، به {language_count} زبان عرضه می‌شود. '
        f'این برنامه توسط {rating_total} کاربر امتیازدهی شده و از ویژگی‌های مخصوص iPad با {ipad_support_count} صفحه پشتیبانی می‌کند.'
    )
    return description

def predict_app_outcome(app_description, tokenizer, model):
    """
    Predict the class (successful/unsuccessful) of an app based on its description using a BERT model.
    """
    inputs = tokenizer(app_description, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

def main():
    st.set_page_config(page_title="App Store Explorer", page_icon="📱", layout="wide")
    
    # بارگذاری مدل‌ها و توکنایزرها
    chat_llm = load_chat_llm()
    tokenizer = load_tokenizer()
    classification_model = load_classification_model()

    # ناوبری و فیلترهای سایدبار
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Data Explorer"])

    # بارگذاری و ترکیب دیتاست‌ها
    app_store_df = pd.read_csv('AppleStore.csv')
    description_df = pd.read_csv('appleStore_description.csv')
    app_store_df['app_desc'] = description_df['app_desc']

    if selected_page == "Home":
        st.title("App Store Data Explorer")

        # تنظیمات فیلتر در سایدبار
        st.sidebar.title("Filters")

        # فیلتر بر اساس ژانر
        selected_genres = st.sidebar.multiselect(
            "Filter by Genre:",
            options=app_store_df["prime_genre"].unique()
        )

        # فیلتر بر اساس ارز
        selected_currencies = st.sidebar.multiselect(
            "Filter by Currency:",
            options=app_store_df["currency"].unique()
        )

        # فیلتر بر اساس محدوده قیمت
        price_min = float(app_store_df["price"].min())
        price_max = float(app_store_df["price"].max())
        selected_price_range = st.sidebar.slider(
            "Filter by Price Range:",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            step=0.01
        )

        # فیلتر بر اساس محدوده اندازه برنامه (بر حسب مگابایت)
        size_mb = app_store_df["size_bytes"] / (1024**2)
        size_min = float(size_mb.min())
        size_max = float(size_mb.max())
        selected_size_range = st.sidebar.slider(
            "Filter by Size (MB):",
            min_value=size_min,
            max_value=size_max,
            value=(size_min, size_max)
        )

        # فیلتر بر اساس محدوده امتیاز کاربران
        rating_min = float(app_store_df["user_rating"].min())
        rating_max = float(app_store_df["user_rating"].max())
        selected_rating_range = st.sidebar.slider(
            "Filter by User Rating:",
            min_value=rating_min,
            max_value=rating_max,
            value=(rating_min, rating_max),
            step=0.1
        )

        # اعمال فیلترها روی دیتافریم
        filtered_df = app_store_df.copy()
        if selected_genres:
            filtered_df = filtered_df[filtered_df["prime_genre"].isin(selected_genres)]
        if selected_currencies:
            filtered_df = filtered_df[filtered_df["currency"].isin(selected_currencies)]
        filtered_df = filtered_df[
            (filtered_df["price"] >= selected_price_range[0]) & (filtered_df["price"] <= selected_price_range[1]) &
            ((filtered_df["size_bytes"] / (1024**2)) >= selected_size_range[0]) & ((filtered_df["size_bytes"] / (1024**2)) <= selected_size_range[1]) &
            (filtered_df["user_rating"] >= selected_rating_range[0]) & (filtered_df["user_rating"] <= selected_rating_range[1])
        ]

        # نمایش داده‌های فیلتر شده
        st.write("### Filtered App Data")
        st.dataframe(filtered_df)

        # انتخاب ردیف برای پردازش
        available_indices = filtered_df.index.tolist()
        if available_indices:
            selected_row_index = st.selectbox("Select a row to process", options=available_indices)
            if st.button("Process Selected Row"):
                selected_row = filtered_df.loc[selected_row_index]

                # تولید توضیحات برنامه
                app_description = generate_app_description(
                    track_name=selected_row["track_name"],
                    genre=selected_row["prime_genre"],
                    currency=selected_row["currency"],
                    price=selected_row["price"],
                    size_bytes=selected_row["size_bytes"],
                    supported_devices=selected_row["sup_devices.num"],
                    app_desc=selected_row["app_desc"],
                    content_rating=selected_row["cont_rating"],
                    language_count=selected_row["lang.num"],
                    rating_total=selected_row["rating_count_tot"],
                    ipad_support_count=selected_row["ipadSc_urls.num"]
                )

                # پیش‌بینی مدل
                predicted_class = predict_app_outcome(app_description, tokenizer, classification_model)
                actual_user_rating = selected_row["user_rating"]

                # تنظیم پیام سیستم بر اساس امتیاز واقعی
                if actual_user_rating >= 4:
                    system_prompt = (
                        'عوامل کلیدی که به محبوبیت این برنامه و استقبال مثبت کاربران از آن کمک کرده‌اند را تجزیه و تحلیل کنید. '
                        'لطفاً پاسخ خود را در حداکثر ۷ خط ارائه دهید و از توضیحات اضافه پرهیز کنید. '
                        'همچنین، فقط از کلمات فارسی استفاده کنید و از به‌کارگیری واژه‌های غیرفارسی خودداری نمایید.'
                    )
                else:
                    system_prompt = (
                        'دلایل خاصی که ممکن است باعث نارضایتی کاربران از این برنامه شود را شناسایی کنید و پیشنهاداتی برای بهبود این مسائل ارائه دهید. '
                        'لطفاً پاسخ خود را در حداکثر ۷ خط ارائه دهید و از توضیحات اضافی خودداری کنید. '
                        'همچنین، فقط از کلمات فارسی استفاده کنید و از به‌کارگیری واژه‌های غیرفارسی خودداری نمایید.'
                    )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=app_description)
                ]
                chatbot_response = chat_llm(messages=messages)

                st.markdown(f'''
                    <div style="direction: rtl; text-align: right; font-size: 18px; font-family: Tahoma;">
                        <strong>پاسخ چت‌بات:</strong> {chatbot_response.content}
                    </div>
                ''', unsafe_allow_html=True)

                st.write(f"Real User Rating: {actual_user_rating}")
                if predicted_class == 1 and actual_user_rating < 4:
                    st.error("Model unsuccessfully predicted a successful outcome for an app with low rating.")
                elif predicted_class == 1 and actual_user_rating >= 4:
                    st.success("Model successfully predicted a successful app.")
                elif predicted_class == 0 and actual_user_rating >= 4:
                    st.error("Model unsuccessfully predicted an unsuccessful outcome for a highly rated app.")
                elif predicted_class == 0 and actual_user_rating < 4:
                    st.success("Model successfully predicted an unsuccessful app.")

                st.write("### Generated Description")
                st.write(app_description)
        else:
            st.warning("No data available for the selected filters.")

    elif selected_page == "Data Explorer":
        st.title("Data Explorer")
        st.write("### Overview of App Data")
        st.dataframe(app_store_df.head(50))

        st.write("### Summary Statistics")
        st.write(app_store_df.describe())

        st.write("### Genre Distribution")
        genre_counts = app_store_df['prime_genre'].value_counts()
        st.bar_chart(genre_counts)

        st.write("### Price Distribution")
        st.line_chart(app_store_df['price'])

        st.write("### User Rating Distribution")
        st.line_chart(app_store_df['user_rating'])

        st.write("### App Size Distribution (MB)")
        st.line_chart(app_store_df['size_bytes'] / (1024**2))

if __name__ == "__main__":
    main()
