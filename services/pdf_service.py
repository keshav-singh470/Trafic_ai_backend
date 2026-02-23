import os
from fpdf import FPDF
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        # Logo placeholder (if needed)
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, '3rd AI - Traffic Violation Report', border=False, ln=True, align='C')
        self.ln(5)
        self.set_draw_color(44, 62, 80)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')

def generate_violation_pdf(job_id, detections, output_path):
    pdf = PDFReport()
    pdf.add_page()
    
    # Metadata Section
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, f'Job ID: {job_id}', ln=True)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 10, f'Total Detections: {len(detections)}', ln=True)
    pdf.ln(5)

    # Table Header
    pdf.set_fill_color(44, 62, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('helvetica', 'B', 10)
    
    headers = [('Plate Number', 45), ('Vehicle Type', 35), ('Color', 25), ('Time', 35), ('Confidence', 30), ('Status', 25)]
    for header, width in headers:
        pdf.cell(width, 10, header, border=1, align='C', fill=True)
    pdf.ln()

    # Table Rows
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('helvetica', '', 9)
    fill = False
    for det in detections:
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        
        pdf.cell(45, 10, str(det.get('plate_number', 'N/A')), border=1, align='C', fill=True)
        pdf.cell(35, 10, str(det.get('vehicle_type', 'unknown')).capitalize(), border=1, align='C', fill=True)
        pdf.cell(25, 10, str(det.get('color', 'unknown')).capitalize(), border=1, align='C', fill=True)
        pdf.cell(35, 10, str(det.get('timestamp', '00:00:00')), border=1, align='C', fill=True)
        pdf.cell(30, 10, f"{det.get('confidence', 0.0):.2f}", border=1, align='C', fill=True)
        status = det.get('status', 'pending')
        pdf.cell(25, 10, status.capitalize(), border=1, align='C', fill=True)
        pdf.ln()
        fill = not fill

    # Section for duplicates or detailed sightings
    if len(detections) > 0:
        pdf.ln(10)
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, 'Detailed Detection Logs (Duplicate Handling)', ln=True)
        pdf.set_font('helvetica', '', 9)
        pdf.multi_cell(0, 5, 'The following list captures all unique tracking events. If a vehicle plate is detected multiple times across different timeframes, each significant instance is recorded below to assist in duplicate analysis.')
        pdf.ln(5)

    pdf.output(output_path)
    return output_path
