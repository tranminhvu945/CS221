import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
from captum.attr import LayerIntegratedGradients
import torch.nn.functional as F
from processors.vietnamese_processor import VietnameseTextPreprocessor 

class PhoBertSentiment(nn.Module):
    def __init__(self):
        super(PhoBertSentiment, self).__init__()
        # Khởi tạo giống hệt lúc train
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.1)
        
        # QUAN TRỌNG: Đổi tên thành classifier để khớp với file .pth
        self.classifier = nn.Linear(768, 2) 

    def forward(self, input_ids, attention_mask):
        # Logic Forward giống hệt lúc train
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Lấy pooler_output (CLS token đã qua xử lý)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits

@st.cache_resource
def load_all_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Preprocessor
    preprocessor = VietnameseTextPreprocessor(vncorenlp_dir='./processors/VnCoreNLP')
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # 3. Load Model
    model = PhoBertSentiment()
    
    # Load weights (đã thêm weights_only=True để tắt cảnh báo)
    try:
        model.load_state_dict(torch.load("phobert_best_model.pth", map_location=device, weights_only=True))
    except RuntimeError as e:
        # Fallback nếu model cũ lưu cả kiến trúc
        st.warning(f"Đang thử load chế độ cũ do lỗi: {e}")
        model.load_state_dict(torch.load("phobert_best_model.pth", map_location=device, weights_only=False))
      
    model.to(device)
    model.eval()
    
    return preprocessor, tokenizer, model, device

# ==========================================
# 2. HÀM GIẢI THÍCH (CAPTUM)
# ==========================================
def visualize_explanation(text, true_label, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_label_idx = torch.max(probs, 1)
        pred_label = pred_label_idx.item()
        conf_score = confidence.item()

    def forward_wrapper(inp, mask):
        return model(inp, mask)

    lig = LayerIntegratedGradients(forward_wrapper, model.phobert.embeddings)
    
    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        baselines=torch.zeros_like(input_ids),
        target=pred_label,
        return_convergence_delta=True,
        n_steps=50
    )

    attributions_sum = attributions.sum(dim=2).squeeze(0)
    attributions_sum = attributions_sum / torch.norm(attributions_sum)
    attr_score = attributions_sum.cpu().detach().numpy()
    
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = [t.replace('_', ' ') for t in raw_tokens]

    real_len = len([t for t in raw_tokens if t != '<pad>'])
    attr_score = attr_score[:real_len]
    tokens = tokens[:real_len]

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(attr_score.reshape(1, -1), cmap='RdYlGn', center=0, 
                annot=False, cbar=True, cbar_kws={'label': 'Tầm quan trọng'}, ax=ax)
    ax.set_xticks(np.arange(len(tokens)) + 0.5)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=12)
    ax.set_yticks([])
    
    label_map = {0: "Non-Constructive", 1: "Constructive"}
    pred_text = label_map[pred_label]
    true_text = label_map[true_label]
    
    if pred_label == true_label:
        status = "CORRECT"
        color = "green"
    else:
        status = "WRONG"
        color = "red"
        
    ax.set_title(f"True: {true_text} | Pred: {pred_text} | {status}", 
                 fontsize=14, fontweight='bold', color=color)
    plt.tight_layout()
    return fig, pred_label, conf_score

# ==========================================
# 3. GIAO DIỆN STREAMLIT
# ==========================================

st.set_page_config(page_title="ViCTSD Analyzer", layout="wide")
st.title("🛡️ ViCTSD: Phân loại bình luận mang tính xây dựng")

try:
    preprocessor, tokenizer, model, device = load_all_resources()
    st.toast("✅ Đã tải Model & VnCoreNLP thành công!", icon="🚀")
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.info("Vui lòng kiểm tra xem thư mục 'VnCoreNLP' đã có đủ file .jar chưa.")
    st.stop()

# Layout Input
col_input, col_label = st.columns([3, 1])

with col_input:
    raw_text = st.text_area("Nhập văn bản:", height=120)

with col_label:
    st.write("### Ground Truth")
    label_option = st.radio(
        "Chọn nhãn thực tế:", 
        ("Non-Constructive (0)", "Constructive (1)"),
        index=0,
        label_visibility="collapsed" 
    )
    true_label_idx = 1 if "Constructive (1)" in label_option else 0

if st.button("🚀 Bắt đầu dự đoán và phân tích", type="primary"):
    if not raw_text.strip():
        st.warning("Vui lòng nhập nội dung.")
    else:
        # --- BƯỚC 1: TIỀN XỬ LÝ ---
        with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
            clean_text = preprocessor.process_text(raw_text)
            status.update(label="Tiền xử lý hoàn tất!", state="complete", expanded=False)

        # Hiển thị so sánh Before/After
        st.subheader("1️⃣ Kết quả Tiền xử lý")
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Văn bản gốc", raw_text, height=130, disabled=True)
        with c2:
            st.text_area("Văn bản đã tiền xử lý", clean_text, height=130, disabled=True)

        st.subheader("2️⃣ Kết quả dự đoán & Giải thích")
        try:
            fig, pred, conf = visualize_explanation(clean_text, true_label_idx, model, tokenizer, device)
            
            # --- HIỂN THỊ KẾT QUẢ ---
            with st.container(border=True):
                # Chia 4 cột
                c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.8, 1])
                
                label_map = {0: "Non-Constructive", 1: "Constructive"}
                true_txt = label_map[true_label_idx]
                pred_txt = label_map[pred]
                
                # CỘT 1: Nhãn Thực Tế
                with c1:
                    st.metric(label="🏷️ Nhãn Thực Tế", value=true_txt)

                # CỘT 2: Nhãn Dự Đoán
                with c2:
                    st.metric(label="🤖 Nhãn Dự Đoán", value=pred_txt)

                # CỘT 3: Trạng Thái
                with c3:
                    if pred == true_label_idx:
                        # delta_color="normal" (màu xanh) cho đúng
                        st.metric(label="Trạng thái", value="Correct", delta="Chính xác", delta_color="normal")
                    else:
                        # delta_color="inverse" (màu đỏ) cho sai
                        st.metric(label="Trạng thái", value="Wrong", delta="Nhầm lẫn", delta_color="inverse")
                
                # CỘT 4: Độ Tin Cậy
                with c4:
                    st.metric(label="📊 Độ Tin Cậy", value=f"{conf:.2%}")

            # Heatmap
            st.write("**Heatmap tầm quan trọng của token:**")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")