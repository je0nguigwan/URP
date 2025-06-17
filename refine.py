import streamlit as st
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
import random
from huggingface_hub import hf_hub_download  # ⬅️ 추가
# 페이지 설정 (가장 위에 있어야 함)
st.set_page_config(page_title="KoBART 욕설 순화기", layout="centered")

# 모델 경로 및 장치 설정
MODEL_PATH = hf_hub_download(
    repo_id="heloolkjdasklfjlasdf/kobart_swear_change",
    filename="06_09kobart_FINAL.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델/토크나이저 불러오기
@st.cache_resource
def load_model_and_tokenizer():
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def refine_text(text):
    input_ids = tokenizer("[순화] " + text, return_tensors="pt", truncation=True).input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=128, num_beams=5)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def fake_profanity_score():
    return round(random.uniform(60.0, 98.0), 2)

# 세션 초기화
if "comments" not in st.session_state:
    st.session_state.comments = [
        {"user": "익명1", "text": "어제 경기 별로였어요"},
        {"user": "익명2", "text": "심판 판정 진짜 최악"},
        {"user": "익명3", "text": "재미는 있었는데 수비가 좀..."},
    ]
if "refined_comment" not in st.session_state:
    st.session_state.refined_comment = None
if "original_comment" not in st.session_state:
    st.session_state.original_comment = None

# ----------------------------------------
# 📢 게시글
# ----------------------------------------
st.markdown("## 📢 게시글")
with st.container():
    st.markdown("""
    <div style='background-color:#2d2d2d; padding:25px; border-radius:12px; border:1px solid #444; color:#f1f1f1'>
        <h3 style='margin-bottom:10px;'>플로리엘 대체 왔다! 한화, 리베라토 영입… "강한 타구 & 넓은 수비 범위 중견수" [공식발표]</h3>
        <p style='color:#aaa; font-size:13px;'>입력 2025.06.17. 오전 10:32 · 수정 2025.06.17. 오전 10:40</p>
    """, unsafe_allow_html=True)

    st.image("https://img7.yna.co.kr/photo/yna/YH/2014/03/20/PYH2014032010070001300_P4.jpg", use_column_width=True)

    st.markdown("""
    <div style='color:#f1f1f1; font-size:16px; line-height:1.7;'>
        <p>[스포츠조선 이종서 기자] 한화 이글스가 발빠르게 대체 외국인 선수 영입을 마쳤다.</p>
        <p>한화는 17일 "우측 새끼손가락 건열골절(뼛조각 생성)로 외국인 재활선수 명단에 오른 에스터반 플로리엘의 대체 외국인 선수로 루이스 리베라토(Luis Liberato, 1995년생, 도미니카공화국)를 영입했다"고 밝혔다.</p>
        <p>"계약 기간은 6주이며, 계약 규모는 총액 약 5만달러"라고 설명했다.</p>
        <p>한화는 "리베라토는 좌투좌타로, 빠른 스윙 스피드를 바탕으로 강한 타구를 생산하는 스타일"이라고 설명했다.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------
# 💬 댓글 출력
# ----------------------------------------
st.markdown("## 💬 댓글")
for c in st.session_state.comments:
    st.markdown(f"""
    <div style='padding:10px 0; border-bottom:1px solid #444;'>
        <strong>{c['user']}</strong><br>
        {c['text']}
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------
# ✍️ 댓글 입력 폼
# ----------------------------------------
st.markdown("## ✍️ 댓글 작성")
with st.form("comment_form"):
    user_input = st.text_area("댓글을 입력하세요", height=80)
    submitted = st.form_submit_button("댓글 등록")

if submitted:
    if user_input.strip():
        refined = refine_text(user_input)
        st.session_state.original_comment = user_input
        st.session_state.refined_comment = refined
    else:
        st.warning("댓글을 입력해주세요.")

# ----------------------------------------
# ⚠️ 순화 안내 + 감지율 + 선택 버튼
# ----------------------------------------
if st.session_state.refined_comment:
    st.markdown("### ⚠️ 이런! 욕설이에요. 다시 한 번 생각해보세요.")
    st.markdown("#### ✨ 순화된 표현 제안:")
    st.success(st.session_state.refined_comment)

    score = fake_profanity_score()
    st.progress(score / 100, text=f"욕설 감지: {score}%")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 순화된 문장으로 댓글 등록"):
            st.session_state.comments.append({
                "user": "익명(나)",
                "text": st.session_state.refined_comment
            })
            st.session_state.refined_comment = None
            st.session_state.original_comment = None
            st.rerun()
    with col2:
        if st.button("✏️ 원문 그대로 등록"):
            st.session_state.comments.append({
                "user": "익명(나)",
                "text": st.session_state.original_comment
            })
            st.session_state.refined_comment = None
            st.session_state.original_comment = None
            st.rerun()
