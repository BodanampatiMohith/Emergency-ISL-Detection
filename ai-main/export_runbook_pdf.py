"""Generate PDF runbook for ISL detection system demo"""

from pathlib import Path
from fpdf import FPDF

def main():
    project_root = Path(__file__).parent
    sections = [
        ("README Extract", project_root / "README.md"),
        ("Complete Detection Guide", project_root / "COMPLETE_DETECTION_GUIDE.md"),
        ("Run Now", project_root / "RUN_NOW.md"),
    ]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for title, path in sections:
        if not path.exists():
            continue

        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(4)

        pdf.set_font("Courier", size=10)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if not text:
                    pdf.ln(5)
                    continue
                safe_text = text.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 5, safe_text)

    output_path = project_root / "ISL_Demo_Runbook.pdf"
    pdf.output(str(output_path))
    print(f"[INFO] PDF generated at {output_path}")


if __name__ == "__main__":
    main()
