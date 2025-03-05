import gradio as gr
import requests
import base64
import io
from PIL import Image
import numpy as np
from zhipuai import ZhipuAI
import os

# Lấy API Key từ biến môi trường
API_KEY = os.getenv("ZHIPUAI_API_KEY", "d659608f7d5d42b1821a9303fc50b618.NesCuIytwpxjxpBJ")
client = ZhipuAI(api_key=API_KEY)

def extract_text(image):
    if isinstance(image, np.ndarray):  # Chuyển NumPy array thành ảnh
        image = Image.fromarray(image)

    # Giới hạn kích thước ảnh
    image = image.resize((min(image.width, 1024), min(image.height, 1024)), Image.Resampling.LANCZOS)

    # Chuyển đổi ảnh sang Base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Gửi yêu cầu đến GLM-4V
    response = client.chat.completions.create(
        model="glm-4v-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": "Hãy trích xuất toàn bộ chữ viết tay trong ảnh, giữ nguyên dấu câu, giữ dấu của từng chữ, khoảng cách dòng, và chính tả như trong ảnh."}
                ]
            }
        ]
    )

    # Xử lý phản hồi từ API
    if response and response.choices:
        text = response.choices[0].message.content
        output_text_file = "output.md"
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(text)

        output_image_file = "output_image.png"
        image.save(output_image_file, format="PNG")

        return text, output_text_file, output_image_file
    else:
        error_msg = f"Lỗi: {str(response)}" if response else "Lỗi: Không kết nối được API."
        return error_msg, None, None

demo = gr.Interface(
    fn=extract_text,
    inputs="image",
    outputs=[
        "text",
        gr.File(label="Tải xuống file văn bản (output.md)"),
    ],
    title="Vietnamese Handwriting OCR",
    description="Upload an image to extract handwritten text using ZhipuAI's GLM-4V model."
)

if __name__ == "__main__":
    demo.launch()
