import markdown
from fpdf import FPDF

# Read the markdown file
with open('explanation.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Convert markdown to html
html = markdown.markdown(text)

# Initialize FPDF
class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "Bovine Vision Project Guide", align="C")
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=11)

# we can use write_html from fpdf2
pdf.write_html(html)

pdf.output('Bovine_Vision_Guide.pdf')
