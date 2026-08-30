import pdfplumber
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslide_report_india_overall.pdf"
)


with pdfplumber.open(PDF_PATH) as pdf:

    # Try extracting table from page 1
    page = pdf.pages[0]

    table = page.extract_table()

    if table:

        print("TABLE FOUND!\n")

        print("Number of rows:", len(table))

        print("\nFirst 10 rows:\n")

        for row in table[:10]:
            print(row)

    else:
        print("No table detected on Page 1.")