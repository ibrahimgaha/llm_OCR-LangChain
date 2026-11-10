import streamlit as st

from ocr import MAX_IMAGE_MB, extract_text, image_to_data_url


def clear_result():
    st.session_state.pop("extracted_text", None)


st.set_page_config(page_title="Image to Text", page_icon="📝")
st.title("Image to Text")
st.caption("Extract text from a PNG or JPEG with NVIDIA Mistral Large 4.")

upload = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    max_upload_size=MAX_IMAGE_MB,
    on_change=clear_result,
)
if upload is None:
    st.stop()

image_bytes = upload.getvalue()
try:
    image_url = image_to_data_url(image_bytes)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.image(image_bytes, caption=upload.name, width="stretch")

if st.button("Extract text", type="primary"):
    clear_result()
    try:
        with st.spinner("Extracting text…"):
            st.session_state.extracted_text = extract_text(image_url)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))

if "extracted_text" in st.session_state:
    text = st.text_area(
        "Extracted text",
        key="extracted_text",
        height=250,
        help="Select the text to copy it, or edit it before downloading.",
    )
    st.download_button("Download text", text, file_name="extracted-text.txt", mime="text/plain")
