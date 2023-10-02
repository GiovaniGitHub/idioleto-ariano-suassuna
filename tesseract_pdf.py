import json
import os
import subprocess

import pytesseract
from pdf2image import convert_from_path


def split_pdf(pdf_path, title, first_page, end_page):
    command = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        f"-dFirstPage={first_page}",
        f"-dLastPage={end_page}",
        f"-sOutputFile=pdfs_splitted/{title}.pdf",
        pdf_path,  # Adjust the file path as needed
    ]

    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def read_metadata(metadata):
    with open(metadata, "r") as f:
        data = json.load(f)
        return data


def pdf_to_text_tesseract(pdf_path, title_file):
    extracted_text = []
    images = convert_from_path(pdf_path)
    for image in images:
        text = pytesseract.image_to_string(image, lang="por")
        extracted_text.append(text)

    full_text = "\n".join(extracted_text)

    with open(f"texts/{title_file}.txt", "w") as f:
        f.write(full_text)


if __name__ == "__main__":
    METADATA = "data/metadata.json"

    metadata = read_metadata(METADATA)
    for elem in metadata:
        if not os.path.isfile(f"texts/{elem.get('titulo')}.pdf"):
            pdf_to_text_tesseract(
                pdf_path=f"pdfs_splitted/{elem.get('titulo')}.pdf",
                title_file=elem.get("titulo"),
            )
