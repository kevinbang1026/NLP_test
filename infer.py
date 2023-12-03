import torch
import streamlit as st
# from kobart import get_kobart_tokenizer
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from load_model import LongformerBartForConditionalGeneration

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    model = LongformerBartForConditionalGeneration.from_pretrained('./kobart_summary')
    # model = LongformerBartForConditionalGeneration.load_from_checkpoint('.checkpoint/last.ckpt')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
st.title("KoBART 요약 Test")
text = st.text_area("뉴스 입력:")

st.markdown("## 뉴스 원문")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## KoBART 요약 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode(text, )
        print(len(input_ids))
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(output)