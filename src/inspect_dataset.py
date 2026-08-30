import pdfplumber
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslide_report_india_overall.pdf"
)


def inspect_pdf():

    print("=" * 60)
    print("LANDSLIDE DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset path: {PDF_PATH}")

    if not PDF_PATH.exists():
        print("\nERROR: Dataset file not found!")
        return

    with pdfplumber.open(PDF_PATH) as pdf:

        print(f"\nTotal pages in PDF: {len(pdf.pages)}")

        print("\n" + "=" * 60)
        print("EXTRACTING TEXT FROM FIRST 3 PAGES")
        print("=" * 60)

        for page_number in range(min(3, len(pdf.pages))):

            page = pdf.pages[page_number]
            text = page.extract_text()

            print(f"\n--- PAGE {page_number + 1} ---\n")

            if text:
                print(text[:2000])
            else:
                print("No text could be extracted.")


if __name__ == "__main__":
    inspect_pdf()