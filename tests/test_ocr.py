import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image
from streamlit.testing.v1 import AppTest

import ocr


def make_image(image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "white").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(("image_format", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_image_data_url(image_format, mime):
    image = make_image(image_format)
    prefix, encoded = ocr.image_to_data_url(image).split(",", 1)
    assert prefix == f"data:{mime};base64"
    assert base64.b64decode(encoded) == image


@pytest.mark.parametrize("image", [b"invalid image", make_image("GIF"), make_image()[:40]])
def test_invalid_image(image):
    with pytest.raises(ValueError, match="valid PNG or JPEG"):
        ocr.image_to_data_url(image)


@pytest.mark.parametrize(
    ("content", "finish_reason", "error"),
    [
        ("Bonjour\nLigne deux", "stop", None),
        ("", "stop", "no extracted text"),
        ("Partial text", "length", "output limit"),
    ],
)
def test_nvidia_image_request(monkeypatch, content, finish_reason, error):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    image_url = ocr.image_to_data_url(make_image())
    calls = []

    def fake_request(session, method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            data = {"data": [{"id": ocr.MODEL_NAME, "owned_by": "mistralai"}]}
        else:
            data = {
                "choices": [
                    {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
                ]
            }
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(data).encode()
        return response

    monkeypatch.setattr(requests.Session, "request", fake_request)
    if error:
        with pytest.raises(RuntimeError, match=error):
            ocr.extract_text(image_url)
    else:
        assert ocr.extract_text(image_url) == content

    method, url, options = calls[-1]
    assert method == "POST"
    assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert options["timeout"] == 90
    payload = options["json"]
    assert payload["model"] == ocr.MODEL_NAME
    # ChatNVIDIA maps its completion limit to the field this endpoint expects.
    assert payload["max_tokens"] == 4096
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ocr.OCR_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    [(requests.exceptions.Timeout(), "timed out"), (requests.exceptions.ConnectionError(), "failed")],
)
def test_api_errors(monkeypatch, failure, message):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(requests.Session, "request", fail)
    with pytest.raises(RuntimeError, match=message):
        ocr.extract_text(ocr.image_to_data_url(make_image()))


def test_missing_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        ocr.extract_text(ocr.image_to_data_url(make_image()))


def test_streamlit_upload_extract_and_replace(monkeypatch):
    monkeypatch.setattr(ocr, "extract_text", lambda image_url: "Bonjour\nLigne deux")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    app.file_uploader[0].upload("sample.png", make_image(), "image/png").run()
    assert not app.exception
    assert app.get("image"), app
    app.button[0].click().run()
    assert not app.exception
    assert app.text_area[0].value == "Bonjour\nLigne deux"
    assert app.get("download_button")[0].label == "Download text"

    app.file_uploader[0].set_value([("new.png", make_image(), "image/png")]).run()
    assert not app.exception
    assert not app.text_area
