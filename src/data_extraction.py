import os
import pdfplumber
import logging
import warnings

warnings.filterwarnings("ignore", message=".*gray non-stroke color.*")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def pdfs_to_single_txt(input_folder, output_file):
    all_text = ""

    # Loop through all PDF files in the folder
    for pdf_file in sorted(os.listdir(input_folder)):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, pdf_file)
            print(f"Processing: {pdf_path}")

            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            all_text += f"\n\n--- {pdf_file} | Page {i+1} ---\n{text}"
                        else:
                            # Fallback to OCR using page image
                            try:
                                page_image = page.to_image(resolution=300)
                                text = page_image.original.text  # This won't work for OCR without pytesseract
                                if text:
                                    all_text += f"\n\n--- {pdf_file} | Page {i+1} (OCR) ---\n{text}"
                            except Exception as e:
                                print(f"Warning: Could not extract text from page {i+1} of {pdf_file}: {e}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    # Save combined text into one file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✅ Combined text saved to: {output_file}")


# Example usage
input_folder = "data/annual_reports"  # Replace with your folder path
output_file = "data/raw/irfc_combined_text.txt"
pdfs_to_single_txt(input_folder, output_file)
