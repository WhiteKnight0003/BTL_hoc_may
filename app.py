import streamlit as st  # Thư viện để xây dựng giao diện web app
import joblib           # Thư viện để tải file model (.pkl) đã huấn luyện
import pandas as pd     # Thư viện xử lý dữ liệu bảng (DataFrame)
import os               # Thư viện hệ điều hành (dùng để kiểm tra file có tồn tại không)

# --- 1. CẤU HÌNH TRANG WEB ---
# Thiết lập tiêu đề tab trình duyệt, icon và bố cục trang
st.set_page_config(
    page_title="AI Phát hiện Tin giả",
    page_icon="🕵️‍♀️",
    layout="centered"
)

# --- 2. HÀM TẢI MODEL (CÓ BỘ NHỚ ĐỆM CACHE) ---
# @st.cache_resource giúp model chỉ cần load 1 lần duy nhất khi khởi động web.
# Nếu không có dòng này, mỗi lần bấm nút, web sẽ load lại model rất chậm.
@st.cache_resource
def load_models():
    # Định nghĩa đường dẫn đến file model
    path_dt = 'model/decision_tree_model.pkl'
    path_svm = 'model/svm_model.pkl'
    
    models = {}
    
    # Kiểm tra xem file Decision Tree có tồn tại không rồi mới load
    if os.path.exists(path_dt):
        models['Decision Tree'] = joblib.load(path_dt)
    else:
        models['Decision Tree'] = None
        st.error(f"Không tìm thấy file: {path_dt}") # Báo lỗi đỏ nếu thiếu file

    # Kiểm tra xem file SVM có tồn tại không rồi mới load
    if os.path.exists(path_svm):
        models['SVM'] = joblib.load(path_svm)
    else:
        models['SVM'] = None
        st.error(f"Không tìm thấy file: {path_svm}")
        
    return models

# --- 3. HÀM HIỂN THỊ KẾT QUẢ ---
# Hàm này nhận vào nhãn dự đoán (prediction) và xác suất (proba) để in ra màn hình
def display_result(prediction, proba):
    # Chuyển xác suất thành dạng phần trăm (Ví dụ: 0.856 -> "85.6%")
    confidence_score = f"{proba * 100:.1f}%"

    # Kiểm tra xem kết quả là Tin giả hay Tin thật
    # Model có thể trả về số 1, chuỗi "Fake" hoặc "fake" tùy cách huấn luyện
    if prediction == 1 or prediction == "Fake" or prediction == "fake":
        # Nếu là Tin Giả: Hiển thị hộp thông báo màu đỏ (st.error)
        st.error(f"🚨 TIN GIẢ (Fake) - Độ chắc chắn: {confidence_score}")
    else:
        # Nếu là Tin Thật: Hiển thị hộp thông báo màu xanh (st.success)
        st.success(f"✅ TIN THẬT (Real) - Độ chắc chắn: {confidence_score}")

# --- 4. HÀM CHÍNH (MAIN) - LOGIC CỦA ỨNG DỤNG ---
def main():
    # Tiêu đề lớn của ứng dụng
    st.title("🕵️‍♀️ Hệ thống Phát hiện Tin giả")
    st.markdown("Nhập tiêu đề và nội dung bài báo để kiểm tra độ tin cậy.")
    st.divider() # Kẻ một đường gạch ngang phân cách

    # Gọi hàm tải model ngay khi vào app
    models = load_models()

    # Tạo một Form để gom nhóm các ô nhập liệu
    # Dùng form giúp trang web không tự chạy lại mỗi khi gõ 1 ký tự
    with st.form("news_form"):
        # Chia giao diện thành 2 cột: Cột 1 nhỏ (1 phần), Cột 2 to (3 phần) - để đẹp hơn (tùy chọn)
        col1, col2 = st.columns([1, 3])
        
        # Ô nhập liệu cho Tiêu đề và Nội dung
        input_title = st.text_area("📝 Tiêu đề bài viết:", height=80, placeholder="Nhập tiêu đề tin tức ở đây...")
        input_text = st.text_area("📄 Nội dung chi tiết:", height=200, placeholder="Nhập nội dung đầy đủ của bài báo...")
        
        # Nút bấm Submit
        submitted = st.form_submit_button("🔍 Kiểm tra ngay", use_container_width=True)

    # Khi người dùng bấm nút "Kiểm tra ngay"
    if submitted:
        # Kiểm tra xem người dùng có bỏ trống ô nào không (.strip() để cắt khoảng trắng thừa)
        if not input_title.strip() or not input_text.strip():
            st.warning("⚠️ Vui lòng nhập đầy đủ cả Tiêu đề và Nội dung!")
        else:
            # Quan trọng: Tạo DataFrame chứa dữ liệu input.
            # Tên cột 'title' và 'text' PHẢI TRÙNG KHỚP với tên cột lúc huấn luyện model
            input_data = pd.DataFrame({
                'title': [input_title],
                'text': [input_text]
            })

            st.subheader("📊 Kết quả phân tích:")
            
            # Chia màn hình kết quả thành 2 cột bằng nhau cho 2 model
            res_col1, res_col2 = st.columns(2)

            # --- XỬ LÝ MODEL DECISION TREE ---
            with res_col1:
                st.info("🌲 Decision Tree Model") # Hộp thông tin màu xanh dương
                if models['Decision Tree']:
                    try:
                        # 1. Dự đoán nhãn (Class): Ra 0 hoặc 1
                        pred = models['Decision Tree'].predict(input_data)[0]
                        
                        # 2. Dự đoán xác suất (Probability): Ra mảng ví dụ [[0.1, 0.9]]
                        # .max() để lấy con số lớn nhất (ví dụ 0.9) làm độ tự tin
                        prob = models['Decision Tree'].predict_proba(input_data).max()
                        
                        # 3. Gọi hàm hiển thị kết quả
                        display_result(pred, prob)
                    except Exception as e:
                        # Bắt lỗi nếu có (ví dụ input data bị lỗi font, model lỗi...)
                        st.error(f"Lỗi: {e}")

            # --- XỬ LÝ MODEL SVM ---
            with res_col2:
                st.info("⚡ SVM Model")
                if models['SVM']:
                    try:
                        # Tương tự như Decision Tree
                        pred = models['SVM'].predict(input_data)[0]
                        
                        # Lưu ý: SVM phải được train với tham số probability=True mới chạy được dòng này
                        prob = models['SVM'].predict_proba(input_data).max()
                        
                        display_result(pred, prob)
                        
                    except Exception as e:
                        st.error(f"Lỗi SVM: {e} (Khả năng do lúc train chưa để probability=True)")

# Điểm bắt đầu chạy chương trình
if __name__ == "__main__":
    main()