import streamlit as st
from PIL import Image
import numpy as np

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def main():

    # Read the processor and model
    processor = TrOCRProcessor.from_pretrained('./processor')
    model = VisionEncoderDecoderModel.from_pretrained('./model')

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    st.title("Image to Text OCR App")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded image')

        with st.spinner("Recognizing text...Please wait."):
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        st.write(f"Recognized text is: {generated_text}")
        
if __name__ == "__main__":
    main()