import pymupdf
import os
from sentence_transformers import SentenceTransformer
import re
import numpy as np


def get_abstract(path):
    doc = pymupdf.open("/data/PDF/2410.02536v2.pdf")
    text = ''
    for page in doc:
        text += page.get_text()

    indexAbstract = text.find("Abstract") + 8
    indexIntroduction = text.find("Introduction")

    text = text[indexAbstract: indexIntroduction]
    print(text)

    filename = "abstract.txt"
    output_dir = "/Users/henry/Documents/Project/PracticeProject/Multimodal-Search/data"
    full_path = os.path.join(output_dir, filename)

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✅ Abstract Text saved to: {full_path}")


# with open('/Users/henry/Documents/Project/PracticeProject/Multimodal-Search/data/abstract.txt', 'r') as file:
#     data = file.read()
#     new_text = re.sub(r"[\n,.]", " ", data)
# abstract_list = new_text.split()
# print(type(abstract_list))

def formListFromtxt(path):
    with open('path', 'r') as file:
        dataRead = file.read()
        newText = re.sub(r"[\n,.]", " ", dataRead)
    return newText.split()

def formDataframe(abstract_list):
    # load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # store abstracts in a list
    # abstract_list = ["abstract 1", "abstract 2"]

    # calculate embeddings
    embeddings = model.encode(abstract_list)

    # Initialize initial value
    finalArray = ""

    # print the result
    for i, emb in enumerate(embeddings):
        # print(f"Embedding for abstract {i + 1}:")
        if isinstance(finalArray, str):
            finalArray = emb
        else:
            finalArray = np.vstack((finalArray, emb))
        print(emb)
        # print(f"Length: {len(emb)}")
        # print()
    return finalArray


# # load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # store abstracts in a list
# abstract_list = ["abstract 1", "abstract 2"]
#
# # calculate embeddings
# embeddings = model.encode(abstract_list)
#
# finalArray = ""
#
# # print the result
# for i, emb in enumerate(embeddings):
#     print(f"Embedding for abstract {i+1}:")
#     if isinstance(finalArray, str):
#         finalArray = emb
#     else:
#         finalArray = np.vstack((finalArray, emb))
#     print(emb)
#     print(f"Length: {len(emb)}")
#     print()
#
# print(len(finalArray))
# print(finalArray.shape)