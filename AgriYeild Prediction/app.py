from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import traceback
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
import os
import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("models/best_crop_yield_model.pkl")

# Load NPK lookup files
state_crop_npk = pd.read_csv("data/state_crop_npk_avg.csv")
crop_npk = pd.read_csv("data/crop_npk_avg.csv")

overall_n = float(state_crop_npk["N"].mean())
overall_p = float(state_crop_npk["P"].mean())
overall_k = float(state_crop_npk["K"].mean())


def get_auto_npk(state: str, crop: str):
    match1 = state_crop_npk[
        (state_crop_npk["State"] == state) &
        (state_crop_npk["crop"] == crop)
    ]
    if not match1.empty:
        row = match1.iloc[0]
        return float(row["N"]), float(row["P"]), float(row["K"])

    match2 = crop_npk[crop_npk["crop"] == crop]
    if not match2.empty:
        row = match2.iloc[0]
        return float(row["N"]), float(row["P"]), float(row["K"])

    return overall_n, overall_p, overall_k


def build_insight_text(rainfall, soil_type, humidity, npk_mode):
    explanation = (
        f"Higher rainfall and suitable soil conditions likely increased the predicted yield. "
        f"The selected soil type ({soil_type}) and humidity level ({humidity}) also contributed to the estimate. "
        f"Nutrient handling mode used: {npk_mode}."
    )

    return {
        "rainfall_impact": "High Impact",
        "temperature_impact": "High Impact",
        "soil_impact": "Medium Impact",
        "npk_impact": "Medium Impact",
        "humidity_impact": "Medium Impact",
        "explanation": explanation
    }


def generate_pdf(data):
    file_path = "prediction_report.pdf"

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=28,
        leftMargin=28,
        topMargin=24,
        bottomMargin=24
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1f5f33"),
        alignment=TA_LEFT,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#4b6b57"),
        alignment=TA_LEFT,
        spaceAfter=8
    )

    section_title_style = ParagraphStyle(
        "SectionTitleStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1f2937")
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=8,
        textColor=colors.HexColor("#1f2937")
    )

    footer_style = ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#4b5563")
    )

    content = []

    logo_path = os.path.join("static", "images", "agri_logo.png")
    banner_path = os.path.join("static", "images", "farm_banner.png")
    input_icon_path = os.path.join("static", "images", "input_icon.png")
    result_icon_path = os.path.join("static", "images", "result_icon.png")
    insight_icon_path = os.path.join("static", "images", "insight_icon.png")

    def safe_image(path, width, height):
        if os.path.exists(path):
            return Image(path, width=width, height=height)
        return Spacer(1, height)

    def section_header(text, icon_path):
        icon = safe_image(icon_path, 18, 18)
        title = Paragraph(text, section_title_style)

        table = Table([[icon, title]], colWidths=[24, 506])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#3d8b3d")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    # Top divider
    top_line = Table([[""]], colWidths=[530])
    top_line.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 1, colors.HexColor("#7aa47d"))
    ]))
    content.append(top_line)
    content.append(Spacer(1, 8))

    # Header
    logo = safe_image(logo_path, 70, 70)
    title_block = Table([[
        Paragraph("AgriYield Predictor Report", title_style)
    ], [
        Paragraph("AI-Powered Crop Yield Analysis", subtitle_style)
    ]], colWidths=[440])

    header_table = Table([[logo, title_block]], colWidths=[80, 450])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 8))

    # Banner
    banner = safe_image(banner_path, 522, 90)

    # Center image properly
    if hasattr(banner, "hAlign"):
        banner.hAlign = "CENTER"

    banner_table = Table([[banner]], colWidths=[530], rowHeights=[98])
    banner_table.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#b7cdb9")),
    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    content.append(banner_table)
    content.append(Spacer(1, 10))

    # Farmer Input Details
    content.append(section_header("Farmer Input Details", input_icon_path))
    content.append(Spacer(1, 8))

    table_data = [["Field", "Value"]]

    table_data += [
        ["State", data['inputs'].get('State', '')],
        ["Crop", data['inputs'].get('Crop', '')],
        ["Rainfall", data['inputs'].get('Rainfall', '')],
        ["Temperature", data['inputs'].get('Temperature', '')],
        ["Humidity", data['inputs'].get('Humidity', '')],
        ["Soil Type", data['inputs'].get('Soil Type', '')],
        ["Land Area", data['inputs'].get('Area', '')],
    ]

    if data["inputs"].get("pH"):
        table_data.append(["pH", data['inputs'].get('pH', '')])
    if data["inputs"].get("N"):
        table_data.append(["N", data['inputs'].get('N', '')])
    if data["inputs"].get("P"):
        table_data.append(["P", data['inputs'].get('P', '')])
    if data["inputs"].get("K"):
        table_data.append(["K", data['inputs'].get('K', '')])

    input_table = Table(table_data, colWidths=[200, 330])

    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f6f3e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#f4f9f4"), colors.white
        ]),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    content.append(input_table)
    content.append(Spacer(1, 12))

    # Prediction Results
   # Prediction Results
    content.append(section_header("Prediction Results", result_icon_path))
    content.append(Spacer(1, 6))

    compact_head = ParagraphStyle(
        "compact_head",
        parent=normal_style,
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    compact_value = ParagraphStyle(
        "compact_value",
        parent=normal_style,
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=colors.white,
        alignment=TA_LEFT,
        leading=17
    )

    # left card
    yield_text = Paragraph(
        f"<b>Predicted Yield</b><br/>{data['yield']} tons/hectare",
        compact_value
    )

    yield_card = Table([[yield_text]], colWidths=[255], rowHeights=[48])
    yield_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#52b437")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # right card
    production_text = Paragraph(
        f"<b>Estimated Production</b><br/>{data['production']} tons",
        compact_value
    )

    production_card = Table([[production_text]], colWidths=[255], rowHeights=[48])
    production_card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#34a6eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    cards_row = Table([[yield_card, production_card]], colWidths=[260, 260])
    cards_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    content.append(cards_row)
    content.append(Spacer(1, 6))

    # AI Insight
    content.append(section_header("AI Insight", insight_icon_path))
    content.append(Spacer(1, 8))

    insight_text = f"• {data['insight']}"
    insight_box = Table([[Paragraph(insight_text, bullet_style)]], colWidths=[530])
    insight_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d1d5db")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    content.append(insight_box)
    content.append(Spacer(1, 10))

    # Footer
    footer_line = Table([[""]], colWidths=[530])
    footer_line.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1"))
    ]))
    content.append(footer_line)
    content.append(Spacer(1, 6))

    footer_table = Table([[
        Paragraph(f"Generated on: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}", footer_style),
        Paragraph("AgriYield Predictor System", footer_style)
    ]], colWidths=[265, 265])

    footer_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    content.append(footer_table)

    doc.build(content)
    return file_path


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "state": request.form.get("state", "").strip(),
        "crop": request.form.get("crop", "").strip(),
        "rainfall": request.form.get("rainfall", "").strip(),
        "temperature": request.form.get("temperature", "").strip(),
        "soil_type": request.form.get("soil_type", "").strip(),
        "humidity": request.form.get("humidity", "").strip(),
        "N": request.form.get("N", "").strip(),
        "P": request.form.get("P", "").strip(),
        "K": request.form.get("K", "").strip(),
        "pH": request.form.get("pH", "").strip(),
        "area": request.form.get("area", "").strip(),
        "unknown_npk": request.form.get("unknown_npk")
    }

    try:
        if form_data["unknown_npk"] == "yes":
            auto_n, auto_p, auto_k = get_auto_npk(form_data["state"], form_data["crop"])
            n_value, p_value, k_value = auto_n, auto_p, auto_k
            form_data["N"] = f"{auto_n:.2f}"
            form_data["P"] = f"{auto_p:.2f}"
            form_data["K"] = f"{auto_k:.2f}"
            npk_mode = "Auto-estimated from historical averages"
        else:
            n_value = float(form_data["N"])
            p_value = float(form_data["P"])
            k_value = float(form_data["K"])
            npk_mode = "User-provided"

        area_value = float(form_data["area"])

        input_df = pd.DataFrame([{
            "State": form_data["state"],
            "crop": form_data["crop"],
            "rainfall": float(form_data["rainfall"]),
            "Avg_Temperature": float(form_data["temperature"]),
            "soil_type": form_data["soil_type"],
            "N": n_value,
            "P": p_value,
            "K": k_value,
            "pH": float(form_data["pH"]),
            "Humidity": float(form_data["humidity"])
        }])

        yield_prediction = float(model.predict(input_df)[0])
        total_production = yield_prediction * area_value

        insights = build_insight_text(
            rainfall=form_data["rainfall"],
            soil_type=form_data["soil_type"],
            humidity=form_data["humidity"],
            npk_mode=npk_mode
        )

        return render_template(
            "index.html",
            prediction_text=f"{yield_prediction:.2f}",
            production_text=f"{total_production:.2f}",
            form_data=form_data,
            insights=insights
        )

    except Exception as e:
        return render_template(
            "index.html",
            error_text=str(e),
            debug_text=traceback.format_exc(),
            form_data=form_data
        )


@app.route("/download", methods=["POST"])
def download():
    data = {
        "inputs": {
            "State": request.form.get("state", ""),
            "Crop": request.form.get("crop", ""),
            "Rainfall": f"{request.form.get('rainfall', '')} mm",
            "Temperature": f"{request.form.get('temperature', '')} °C",
            "Humidity": f"{request.form.get('humidity', '')} %",
            "Soil Type": request.form.get("soil_type", ""),
            "pH": request.form.get("pH", ""),
            "N": request.form.get("N", ""),
            "P": request.form.get("P", ""),
            "K": request.form.get("K", ""),
            "Area": f"{request.form.get('area', '')} hectares",
        },
        "yield": f"{request.form.get('prediction', '')}",
        "production": f"{request.form.get('production', '')}",
        "insight": request.form.get(
            "insight",
            "Prediction based on weather conditions, soil properties, and nutrient levels."
        )
    }

    file_path = generate_pdf(data)
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False)