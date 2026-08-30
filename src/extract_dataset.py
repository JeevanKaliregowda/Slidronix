import pdfplumber
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslide_report_india_overall.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_raw.csv"
)


# --------------------------------------------------
# EXPECTED COLUMNS
# --------------------------------------------------

EXPECTED_COLUMNS = [
    "S.No",
    "Latitude",
    "Longitude",
    "Slide_Name",
    "State",
    "District",
    "Subdivision Or Taluk",
    "Material Involved",
    "Movement Type",
    "Initiation_Year",
    "History_date"
]


# --------------------------------------------------
# EXTRACTION
# --------------------------------------------------

all_rows = []

print("=" * 60)
print("STARTING LANDSLIDE DATASET EXTRACTION")
print("=" * 60)

with pdfplumber.open(PDF_PATH) as pdf:

    total_pages = len(pdf.pages)

    print(f"\nTotal pages: {total_pages}\n")

    for page_number, page in enumerate(pdf.pages, start=1):

        table = page.extract_table()

        if not table:
            print(f"Page {page_number}: No table found")
            continue

        page_rows_added = 0

        for row in table:

            # Skip empty rows
            if not row:
                continue

            # Skip title row
            if row[0] and "LANDSLIDE INVENTORY" in row[0]:
                continue

            # Skip repeated header rows
            if row[0] == "S.No":
                continue

            # Keep only rows with expected number of columns
            if len(row) != len(EXPECTED_COLUMNS):
                continue

            all_rows.append(row)
            page_rows_added += 1

        if page_number % 25 == 0 or page_number == total_pages:
            print(
                f"Processed page {page_number}/{total_pages} "
                f"| Rows collected: {len(all_rows)}"
            )


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(
    all_rows,
    columns=EXPECTED_COLUMNS
)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXTRACTION COMPLETED")
print("=" * 60)

print(f"\nTotal records extracted: {len(df)}")
print(f"Output saved to:\n{OUTPUT_PATH}")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)