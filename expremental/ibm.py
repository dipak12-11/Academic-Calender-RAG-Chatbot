import os

# Fix for Windows symlink privilege error
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain_docling.loader import DoclingLoader

FILE_PATH = "https://arxiv.org/pdf/2408.09869"

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()
for d in docs[:3]:
    print(f"- {d.page_content=}")