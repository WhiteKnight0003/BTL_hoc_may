import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="AI Phát hiện Tin giả",
    page_icon="🕵️‍♀️",
    layout="centered"
)

@st.cache_resource
def load_models():
    path_dt = 'model/decision_tree_model.pkl'
    path_svm = 'model/svm_model.pkl'
    
    models = {}
    if os.path.exists(path_dt):
        models['Decision Tree'] = joblib.load(path_dt)
    else:
        models['Decision Tree'] = None
        st.error(f"Không tìm thấy file: {path_dt}")

    if os.path.exists(path_svm):
        models['SVM'] = joblib.load(path_svm)
    else:
        models['SVM'] = None
        st.error(f"Không tìm thấy file: {path_svm}")
        
    return models

def display_result(prediction):
    label_map = {
        1: "TIN GIẢ (Fake)",  
        0: "TIN THẬT (Real)", 
        "Fake": "TIN GIẢ (Fake)",
        "Real": "TIN THẬT (Real)",
        "fake": "TIN GIẢ (Fake)",
        "real": "TIN THẬT (Real)"
    }
    
    result_text = label_map.get(prediction, str(prediction))

    is_fake = prediction in [1, "Fake", "fake"] 
    
    if is_fake:
        st.error(f"🚨 {result_text}")
    else:
        st.success(f"✅ {result_text}")

def main():
    st.title("🕵️‍♀️ Hệ thống Phát hiện Tin giả")
    st.markdown("Nhập tiêu đề và nội dung bài báo để kiểm tra độ tin cậy.")
    st.divider()

    models = load_models()

    with st.form("news_form"):
        col1, col2 = st.columns([1, 3])
        
        input_title = st.text_area("📝 Tiêu đề bài viết:", height=80, placeholder="Nhập tiêu đề tin tức ở đây...")
        input_text = st.text_area("📄 Nội dung chi tiết:", height=200, placeholder="Nhập nội dung đầy đủ của bài báo...")
        
        submitted = st.form_submit_button("🔍 Kiểm tra ngay", use_container_width=True)

    if submitted:
        if not input_title.strip() or not input_text.strip():
            st.warning("⚠️ Vui lòng nhập đầy đủ cả Tiêu đề và Nội dung!")
        else:
            input_data = pd.DataFrame({
                'title': [input_title],
                'text': [input_text]
            })

            st.subheader("📊 Kết quả phân tích:")
            
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.info("🌲 Decision Tree Model")
                if models['Decision Tree']:
                    try:
                        prediction = models['Decision Tree'].predict(input_data)[0]
                        display_result(prediction)
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                else:
                    st.text("Model chưa được tải.")

            with res_col2:
                st.info("⚡ SVM Model")
                if models['SVM']:
                    try:
                        prediction = models['SVM'].predict(input_data)[0]
                        display_result(prediction)
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                else:
                    st.text("Model chưa được tải.")



if __name__ == "__main__":
    main()

# pip install streamlit joblib pandas scikit-learn
# streamlit run app.py