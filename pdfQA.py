import fitz  # PyMuPDF
from transformers import pipeline

# pdf_document = "sample2.pdf"
# document = fitz.open(pdf_document)

# pdf_text = ""
# for page_num in range(len(document)):
#     page = document.load_page(page_num)
#     pdf_text += page.get_text()


# pdf_text = pdf_text.replace("Professional IEC 60243 test machine manufacturer: Email: server@kepin17.com","")
# print(pdf_text)


# file = open("sample2.txt","w",encoding="utf-8")
# file.writelines(pdf_text)

file2 = open("sample2.txt","r",encoding="utf-8")
# print(file2.readlines())
text = file2.readlines()
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
question = "What is this pdf about?"

# Get the answer
result = qa_pipeline(question=question, context=text)
print(result["answer"])