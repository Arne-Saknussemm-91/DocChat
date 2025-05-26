from pdfplumber import open as pdf_open

def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with pdf_open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() + "\n"
    return extracted_text.strip()

def extract_text_from_scanned_pdf(pdf_path):
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(pdf_path)
    extracted_text = ""
    for image in images:
        extracted_text += pytesseract.image_to_string(image) + "\n"
    return extracted_text.strip()