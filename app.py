import os
from typing import Any

import torch
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path

# Настройка устройства
device = get_torch_device('auto')
model_name = 'vidore/colqwen2-v1.0'
# local_model_path = "/path/to/your/local/model"
# TODO: remove it
local_model_path = "/home/meno/models/colqwen2-v1.0"

# Модели и процессоры
text_embedder = SentenceTransformer('intfloat/multilingual-e5-large')
lora_config = LoraConfig.from_pretrained(local_model_path)
processor_retrieval = ColQwen2Processor.from_pretrained(local_model_path)
processor_generation = Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path)


class ColQwen2ForRAG(ColQwen2):
    """
    ColQwen2 model implementation that can be used both for retrieval and generation.
    Allows switching between retrieval and generation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_retrieval_enabled = True

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
        or ColQwen2.forward for retrieval based on the current mode.
        """
        if self.is_retrieval_enabled:
            return ColQwen2.forward(self, *args, **kwargs)
        else:
            return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate text using Qwen2VLForConditionalGeneration.generate.
        """
        if not self.is_generation_enabled:
            raise ValueError(
                'Set the model to generation mode by calling `enable_generation()` before calling `generate()`.')
        return super().generate(*args, **kwargs)

    @property
    def is_retrieval_enabled(self) -> bool:
        return self._is_retrieval_enabled

    @property
    def is_generation_enabled(self) -> bool:
        return not self.is_retrieval_enabled

    def enable_retrieval(self) -> None:
        """
        Switch to retrieval mode.
        """
        self.enable_adapters()
        self._is_retrieval_enabled = True

    def enable_generation(self) -> None:
        """
        Switch to generation mode.
        """
        self.disable_adapters()
        self._is_retrieval_enabled = False


model = ColQwen2ForRAG.from_pretrained(local_model_path, torch_dtype=torch.float16, device_map=device)


# Функции обработки файлов
def process_pdf(pdf_path):
    """Извлекает текст и изображения из PDF."""
    reader = PdfReader(pdf_path)
    page_texts = [page.extract_text() for page in reader.pages]
    images = convert_from_path(pdf_path)
    return images, page_texts


def process_doc(doc_path):
    """Извлекает текст из .docx файла."""
    doc = Document(doc_path)
    texts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return None, texts  # .doc файлы не имеют встроенных изображений


def validate_file_format(file):
    """Проверяет формат файла."""
    allowed_extensions = ["pdf", "doc", "docx"]
    filename = file.name.lower()
    extension = filename.split(".")[-1]
    return extension in allowed_extensions, extension


# Функции обработки PDF
def get_pdf_images(pdf_name):
    """Получает текст и изображения из PDF."""
    reader = PdfReader(pdf_name)
    page_texts = [page.extract_text() for page in reader.pages]
    images = convert_from_path(pdf_name)
    return images, page_texts


def scale_image(image: Image.Image, new_height: int = 1024):
    """Масштабирует изображение, сохраняя пропорции."""
    width, height = image.size
    new_width = int(new_height * width / height)
    return image.resize((new_width, new_height))


# Интерфейс Streamlit
st.title("PDF Knowledge Retrieval System")

# Загрузка PDF
uploaded_files = st.file_uploader("Upload PDF or DOC files", accept_multiple_files=True)
document_data = {}

if uploaded_files:
    for file in uploaded_files:
        is_valid, extension = validate_file_format(file)
        if not is_valid:
            st.error(f"Unsupported file format: {file.name}. Please upload only PDF or DOC files.")
            continue

        if extension == "pdf":
            images, texts = process_pdf(file)
        elif extension in ["doc", "docx"]:
            images, texts = process_doc(file)
        else:
            continue  # Это не должно происходить, так как проверка выше фильтрует недопустимые форматы

        document_data[file.name] = {"images": images, "texts": texts}
    if document_data:
        st.success(f"Successfully uploaded and processed {len(document_data)} document(s).")

# Поиск
query = st.text_input("Enter your query")
num_results = st.slider("Number of retrieved pages", 1, 10, 3)

query = st.text_input("Enter your query")
num_results = st.slider("Number of retrieved pages", 1, 10, 3)

if st.button("Search"):
    if document_data:
        for doc_name, data in document_data.items():
            page_texts = data["texts"]
            page_images = data["images"]

            # Ретривер
            query_embedding = text_embedder.encode([f"query: {query}"])
            page_embeddings = text_embedder.encode([f"passage: {text}" for text in page_texts])
            similarities = query_embedding @ page_embeddings.T
            top_indices = sorted(range(len(similarities[0])), key=lambda i: -similarities[0][i])[:num_results]

            retrieved_texts = [page_texts[i] for i in top_indices]
            retrieved_images = [scale_image(page_images[i], 512) for i in top_indices] if page_images else []

            # Генерация ответа
            united_retrieved_doc = "\n".join(retrieved_texts)
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text",
                         "text": "Вы - эксперт по российской промышленности. Отвечайте на вопросы точно и полно."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Документ:\n{united_retrieved_doc}"},
                        {"type": "text", "text": f"Вопрос:\n{query}"}
                    ]
                }
            ]

            image_inputs = processor_retrieval.process_images(retrieved_images).to(device) if retrieved_images else None
            inputs_generation = processor_generation(
                text=[processor_generation.apply_chat_template(conversation, add_generation_prompt=True)],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            model.enable_generation()
            with torch.no_grad():
                output_ids = model.generate(**inputs_generation, max_new_tokens=512, do_sample=True)

            generated_text = processor_generation.batch_decode(output_ids, skip_special_tokens=True)[0]

            # Результаты
            st.write(f"**Document:** {doc_name}")
            st.write(f"**Query:** {query}")
            st.write(f"**Generated Answer:** {generated_text}")
            for img in retrieved_images:
                st.image(img, use_column_width=True)
    else:
        st.error("Please upload and process a document before searching.")
