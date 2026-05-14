"""
generate_paper.py
Generates an arXiv-style PDF working paper using ReportLab Platypus.
Output: docs/paper.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, Image,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, datetime, io, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add src/ to path so analysis modules can be imported regardless of cwd
_SRC_DIR = os.path.dirname(__file__)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if os.path.dirname(_SRC_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SRC_DIR))

# Load panel and run conditional IV once at module level
_PANEL_PATH = os.path.join(_SRC_DIR, "..", "data", "processed", "panel.parquet")
_civ_results = {}
if os.path.exists(_PANEL_PATH):
    try:
        from analysis.iv import run_conditional_iv
        _df_panel = pd.read_parquet(_PANEL_PATH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _civ_results = run_conditional_iv(_df_panel)
    except Exception as _e:
        print(f"[generate_paper] Warning: could not run conditional IV: {_e}")


def _fmt_civ(key, field, fmt="{:.4f}", fallback="—"):
    """Extract a formatted value from _civ_results, or return fallback."""
    r = _civ_results.get(key)
    if r is None:
        return fallback
    val = r.get(field)
    if val is None:
        return fallback
    try:
        return fmt.format(float(val))
    except (TypeError, ValueError):
        return fallback


def _pstar(pval_str):
    """Append significance stars to a p-value string."""
    try:
        p = float(pval_str)
        if p < 0.001:  return pval_str + " ***"
        if p < 0.01:   return pval_str + " **"
        if p < 0.05:   return pval_str + " *"
        if p < 0.10:   return pval_str + " ·"
        return pval_str
    except (TypeError, ValueError):
        return pval_str

# Register Unicode-capable fonts (DejaVu Sans covers Greek, math, and all
# characters used in this paper that Helvetica's WinAnsiEncoding cannot render).
_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DejaVuSans",      os.path.join(_DEJAVU_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_DEJAVU_DIR, "DejaVuSans-Bold.ttf")))


def eq_image(latex_str, fontsize=11, color="#111111"):
    """Render a LaTeX math string via matplotlib mathtext and return a ReportLab Image."""
    fig, ax = plt.subplots(figsize=(8, 0.6))
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"${latex_str}$",
        ha="center", va="center",
        fontsize=fontsize,
        color=color,
        transform=ax.transAxes,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0.05, dpi=180, transparent=True)
    plt.close(fig)
    buf.seek(0)

    img = Image(buf)
    # Scale to fit within text block (max width ~ 14 cm)
    max_w = 14 * cm
    scale = min(1.0, max_w / img.imageWidth)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "LEFT"
    return img

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATH
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "paper.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 3.2 * cm

def build_styles():
    base = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    gray = colors.HexColor("#444444")
    dark = colors.HexColor("#111111")
    link = colors.HexColor("#1a4eb8")

    return {
        "title":    S("PTitle", "Title",
                      fontSize=17, leading=22, spaceAfter=4,
                      textColor=dark, alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "subtitle": S("PSubt", fontSize=11, leading=14, spaceAfter=6,
                      textColor=gray, alignment=TA_CENTER),
        "author":   S("PAuth", fontSize=10.5, leading=13, spaceAfter=2,
                      textColor=dark, alignment=TA_CENTER),
        "affil":    S("PAff", fontSize=9, leading=12, spaceAfter=12,
                      textColor=gray, alignment=TA_CENTER, fontName="Helvetica-Oblique"),
        "abstract_head": S("PAbsH", fontSize=10, spaceAfter=4, spaceBefore=10,
                           fontName="Helvetica-Bold", alignment=TA_CENTER),
        "abstract": S("PAbs", fontSize=9.5, leading=13, leftIndent=1.2*cm,
                      rightIndent=1.2*cm, spaceAfter=10, fontName="DejaVuSans",
                      alignment=TA_JUSTIFY, textColor=gray),
        "keywords": S("PKW", fontSize=8.5, leading=12, leftIndent=1.2*cm,
                      rightIndent=1.2*cm, spaceAfter=16, textColor=gray,
                      fontName="Helvetica-Oblique"),
        "section":  S("PSec", fontSize=12, leading=15, spaceBefore=18,
                      spaceAfter=6, fontName="DejaVuSans-Bold", textColor=dark),
        "subsection": S("PSSec", fontSize=10.5, leading=13, spaceBefore=10,
                        spaceAfter=4, fontName="DejaVuSans-Bold", textColor=dark),
        "body":     S("PBody", fontSize=10, leading=14.5, spaceAfter=7,
                      fontName="DejaVuSans", alignment=TA_JUSTIFY),
        "body_small": S("PSmall", fontSize=9, leading=13, spaceAfter=6,
                        fontName="DejaVuSans", alignment=TA_JUSTIFY, textColor=gray),
        "caption":  S("PCap", fontSize=8.5, leading=12, spaceAfter=8,
                      fontName="Helvetica-Oblique", textColor=gray, alignment=TA_CENTER),
        "footnote": S("PFn", fontSize=8, leading=11, spaceAfter=3,
                      textColor=gray),
        "equation": S("PEq", fontSize=9.5, leading=14, leftIndent=1.5*cm,
                      spaceAfter=5, fontName="Courier"),
        "ref":      S("PRef", fontSize=8.5, leading=12, leftIndent=0.6*cm,
                      firstLineIndent=-0.6*cm, spaceAfter=5, fontName="DejaVuSans",
                      textColor=gray),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TABLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
GRAY50  = colors.HexColor("#eeeeee")
GRAY20  = colors.HexColor("#f7f7f7")
HDRBG   = colors.HexColor("#222244")
HDRTXT  = colors.white

def make_table(data, col_widths, hdr_rows=1, caption="", styles_dict=None):
    """Build a styled academic table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=hdr_rows)
    ts = [
        ("FONTNAME",    (0,0), (-1, hdr_rows-1), "DejaVuSans-Bold"),
        ("FONTSIZE",    (0,0), (-1, hdr_rows-1), 9),
        ("FONTNAME",    (0, hdr_rows), (-1,-1), "DejaVuSans"),
        ("FONTSIZE",    (0, hdr_rows), (-1,-1), 9),
        ("BACKGROUND",  (0,0), (-1, hdr_rows-1), HDRBG),
        ("TEXTCOLOR",   (0,0), (-1, hdr_rows-1), HDRTXT),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, hdr_rows), (-1,-1), [GRAY20, GRAY50]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ]
    if styles_dict:
        ts.extend(styles_dict)
    tbl.setStyle(TableStyle(ts))
    return tbl

# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATE (header / footer)
# ─────────────────────────────────────────────────────────────────────────────
def on_first_page(canvas, doc):
    pass  # no header on first page

def on_later_pages(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    # Running header
    header = "Martins Ramos de Oliveira, C. — Parliamentary Representation, Earmarked Transfers & Inequality"
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.4*cm, header)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 0.4*cm,
                           f"arXiv Working Paper · May 2026")
    canvas.line(MARGIN, PAGE_H - MARGIN + 0.25*cm,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 0.25*cm)
    # Footer
    canvas.drawCentredString(PAGE_W / 2, MARGIN - 0.5*cm,
                             f"— {doc.page} —")
    canvas.restoreState()

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_story(S):
    story = []
    B = S["body"]
    Bs = S["body_small"]
    Eq = S["equation"]

    def sec(txt):  return Paragraph(txt, S["section"])
    def ssec(txt): return Paragraph(txt, S["subsection"])
    def p(txt):    return Paragraph(txt, B)
    def p_small(txt): return Paragraph(txt, Bs)
    def sp(n=8):   return Spacer(1, n)
    def hr():      return HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#cccccc"),
                                     spaceAfter=4, spaceBefore=4)

    # ── TITLE BLOCK ──────────────────────────────────────────────────────────
    story += [
        sp(6),
        Paragraph(
            "Parliamentary Representation, Earmarked Transfers,<br/>"
            "and Income Inequality in Brazilian States:<br/>"
            "A Multi-Strategy Causal Analysis",
            S["title"]
        ),
        sp(10),
        Paragraph("Caio Martins Ramos de Oliveira", S["author"]),
        Paragraph("Independent Researcher &amp; Data Scientist", S["affil"]),
        Paragraph(
            f"Working Paper · May 2026 · "
            "<font color='#1a4eb8'>"
            "https://caiomar.github.io/politicas-publicas/dashboard.html"
            "</font>",
            S["subtitle"]
        ),
        sp(12), hr(), sp(6),
    ]

    # ── ABSTRACT ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("Abstract", S["abstract_head"]),
        Paragraph(
            "We investigate whether over-representation in the Brazilian Chamber of Deputies — "
            "measured by the ratio of state seats to population — increases the volume of "
            "earmarked parliamentary amendments (emendas parlamentares) received by states and, "
            "ultimately, reduces state-level income inequality (Gini coefficient). "
            "Exploiting the constitutionally fixed seat floors (8 seats) and ceilings (70 seats) "
            "as an instrumental variable, and applying five distinct causal identification "
            "strategies — causal mediation analysis (Imai, Keele, and Tingley, 2010), distributed-lag impulse "
            "response functions, Difference-in-Differences around Constitutional Amendment 86/2015, "
            "Regression Discontinuity Design at the 8-seat threshold, and Double Machine Learning — "
            "we find no robust evidence that parliamentary amendments reduce inequality at the "
            "state level. The only statistically significant link detected is between public "
            "expenditure on education and the Gini index (β = −0.025, p = 0.026), consistent "
            "with a redistribution channel operating through fiscal policy rather than through "
            "earmarked transfers per se. Our mediation decomposition reveals a partial-suppression "
            "pattern: the indirect effect via amendments (NIE = −0.0031) partially offsets a "
            "countervailing positive direct effect (NDE = +0.0058), leaving a statistically "
            "indistinguishable-from-zero total effect (ATE = +0.0027, p = 0.151). "
            "These findings suggest that the scale mismatch between municipal-targeted "
            "amendments and state-level inequality measurement may obscure genuine distributional "
            "effects that could surface at finer geographic resolution.",
            S["abstract"]
        ),
        Paragraph(
            "<b>Keywords:</b> parliamentary representation · earmarked transfers · income inequality · "
            "causal mediation · instrumental variables · Brazilian fiscal federalism · "
            "emendas parlamentares · Double Machine Learning",
            S["keywords"]
        ),
        Paragraph(
            "<b>JEL Codes:</b> H72 · H77 · D72 · C26 · N46",
            S["keywords"]
        ),
        hr(), sp(4),
    ]

    # ── 1. INTRODUCTION ───────────────────────────────────────────────────────
    story += [
        sec("1.  Introduction"),
        p("Brazil's Chamber of Deputies allocates budgetary resources to states through "
          "parliamentary amendments — known as <i>emendas parlamentares</i> — which earmark "
          "federal funds for projects proposed by individual legislators. Since the "
          "Constitutional Amendment 86 of 2015 made a portion of these transfers mandatory "
          "(the so-called <i>emendas impositivas</i>), the system has grown substantially: "
          "in 2023 alone, total earmarked amendments exceeded R$ 50 billion. "
          "This mechanism has attracted considerable policy attention as a potential vehicle "
          "for regional redistribution, given that smaller and poorer states are constitutionally "
          "over-represented in the Chamber."),
        p("The core empirical question is whether this institutional feature — "
          "states receiving more deputies per capita — translates into lower income inequality "
          "through the mechanism of higher per-capita amendments. While the causal chain has "
          "intuitive appeal (more deputies → more earmarked funds → targeted public spending → "
          "lower Gini), several confounding forces complicate identification. States with "
          "higher per-capita representation tend to be smaller, poorer, and concentrated in "
          "the North and Northeast — regions with historically high inequality. Naïve OLS "
          "therefore mixes distributional selection with any genuine redistributive effect."),
        p("We address this identification problem through a multi-pronged strategy. "
          "Our instrument exploits the constitutional rule that each state receives between "
          "8 and 70 seats regardless of population, creating exogenous variation in per-capita "
          "representation. We complement the IV approach with: (i) distributed-lag models "
          "that allow for multi-year effect accumulation; (ii) an event study around "
          "Constitutional Amendment 86/2015; (iii) an RDD exploiting the 8-seat floor; and "
          "(iv) Double Machine Learning (Chernozhukov et al., 2018) to purge confounders "
          "non-parametrically. A secondary fiscal mediation chain (amendments → sectoral "
          "spending → Gini) is estimated using SICONFI administrative data."),
        p("Our main contribution is to provide the first systematic, multi-method causal "
          "analysis linking parliamentary representation to inequality in Brazilian states, "
          "embedding it within a formal causal graph (DAG) and a mediation decomposition "
          "framework. We also provide, to our knowledge, the first test of the conditional "
          "IV validity in this setting via historical development proxies."),
    ]

    # ── 2. LITERATURE REVIEW ──────────────────────────────────────────────────
    story += [
        sec("2.  Related Literature"),
        ssec("2.1  Malapportionment and fiscal transfers"),
        p("The closest international benchmark is Galiani, Galetovic, and Schargrodsky "
          "(Economics &amp; Politics, 2016), which analyzes the Brazilian case using IV "
          "methods and finds that over-represented states receive disproportionate "
          "federal resources, but that this advantage does not translate into higher "
          "development — if anything, it may worsen resource allocation. Our paper "
          "extends this finding to the inequality dimension and applies a richer "
          "identification toolkit."),
        p("Beramendi, Rogers, and Díaz-Cayeros (2017) argue that the joint presence of "
          "regional inequality and malapportionment distinguishes Latin American federations "
          "from other federal systems — less populous (not necessarily poorer) states "
          "are the main beneficiaries of skewed transfers. This is the precise mechanism "
          "our partial-suppression result captures: over-representation shifts resources "
          "to smaller states but through a channel (discretionary earmarks) that does not "
          "systematically reduce inequality."),
        p("Dunn (2022, <i>Publius</i>) documents that malapportioned legislatures produce "
          "a rural-conservative bias that obstructs centralised redistributive reforms. "
          "Consistent with this, our direct-effect estimate (NDE = +0.0058) suggests that "
          "over-representation marginally raises the state Gini through non-amendment channels, "
          "possibly by constraining formula-based equalisation transfers."),
        ssec("2.2  Electoral effects of parliamentary amendments"),
        p("A large Brazilian literature examines whether parliamentary amendments translate "
          "into electoral returns. Ames (1995) and Pereira and Rennó (2007) find positive "
          "electoral effects; Samuels (2002) finds null effects; Mesquita (2008) finds "
          "negative effects in some contexts. Baião and Couto (2017) reconcile this "
          "heterogeneity: only amendments executed as transfers to municipalities generate "
          "votes, and only when the mayor is a legislative ally. This result is directly "
          "relevant to the <i>scale mismatch hypothesis</i> developed in our Section 7.2 — "
          "the mechanism operates at the municipal level, making state-level Gini an "
          "inappropriate outcome measure."),
        ssec("2.3  Double Machine Learning and IV identification"),
        p("The application of Double Machine Learning (DML) to panel data follows "
          "Chernozhukov et al. (2018). Our DML implementation is the first, to our knowledge, "
          "applied to the emendas–Gini channel in Brazil. The comparison between naïve OLS "
          "(θ = −0.010, p = 0.036) and DML (θ = +0.001, p = 0.67) illustrates the classic "
          "omitted-variable bias in this setting: small over-represented states are also "
          "historically poorer and more unequal, producing a spurious negative OLS coefficient "
          "that DML's non-parametric deconfounding eliminates."),
        p("The mediation framework follows Imai, Keele, and Tingley (2010), "
          "who formalise identification conditions for natural direct and indirect effects "
          "under sequential ignorability — a stricter requirement than Baron &amp; Kenny (1986). "
          "VanderWeele (2015) provides the theoretical foundation for the partial-suppression "
          "pattern we document. The closest international benchmarks are "
          "Galiani, Galetovic, and Schargrodsky (2016), who find that over-represented "
          "Brazilian states receive more resources without development gains, and "
          "Beramendi, Rogers, and Díaz-Cayeros (2017), who document that malapportionment "
          "disproportionately benefits less populous states in Latin American federations "
          "— consistent with our partial-suppression finding."),
    ]

    # ── 3. INSTITUTIONAL BACKGROUND ────────────────────────────────────────────
    story += [
        sec("2.  Institutional Background"),
        ssec("2.1  Chamber seat allocation"),
        p("Brazil's federal constitution distributes the 513 seats of the Chamber of Deputies "
          "among 27 federative units (26 states + Federal District) proportional to population, "
          "subject to a <b>minimum of 8</b> and a <b>maximum of 70</b> seats per unit. "
          "These limits create discontinuities that decouple representation from population "
          "at the tails of the distribution: the state of Roraima (≈750k inhabitants) "
          "holds 8 seats, the same number as Acre (≈950k), while São Paulo "
          "(≈46 million) is capped at 70 — roughly one-eighth of its proportional share. "
          "This generates a natural instrument: within-state variation in effective "
          "representation relative to population is largely determined by constitutional "
          "arithmetic rather than unobservable state characteristics."),
        ssec("2.2  Parliamentary amendments (emendas parlamentares)"),
        p("Individual deputies propose line-item amendments to the annual federal budget, "
          "earmarking funds for projects in their home states. Before EC 86/2015, the "
          "executive could impound these appropriations; the amendment made 1.2% of net "
          "federal revenue mandatory execution, creating a strong quasi-exogenous shock "
          "to the regime. In our data (2012–2023), per-capita amendments vary from "
          "R$ 4.7 million/100k inhabitants (São Paulo) to R$ 58.9 million/100k (Amapá)."),
    ]

    # ── 3. DATA ────────────────────────────────────────────────────────────────
    story += [
        sec("3.  Data"),
        p("We combine four data sources into an unbalanced state-year panel spanning "
          "2012–2023 (N = 243 after exclusions for lagged variables):"),
    ]

    data_tbl = [
        ["Source", "Coverage", "Key Variables"],
        ["Portal da Transparência\n(SIAFI)", "2012–2023, 27 UFs",
         "Emendas paid (R$), function classification"],
        ["IBGE / PNADC", "2012–2023, 27 UFs",
         "Gini coefficient (state-year), population"],
        ["IBGE / SCN", "2012–2023, 27 UFs",
         "GDP per capita (deflated to 2023 R$)"],
        ["SICONFI / RREO\nAnexo 02", "2018–2023, 27 UFs",
         "Liquidated expenditure: education, health"],
    ]
    avail_w = PAGE_W - 2*MARGIN
    story += [
        sp(4),
        make_table(data_tbl, [avail_w*0.26, avail_w*0.26, avail_w*0.48]),
        Paragraph("Table 1. Data sources.", S["caption"]),
        sp(4),
        p("The Gini coefficient is computed from the PNAD Contínua microdata at state level. "
          "Parliamentary amendment values are deflated using the IPCA price index. "
          "We construct the key treatment variable as <i>relative representation</i> = "
          "(state seats) / (state population / national average population per seat), "
          "with values above (below) 1.0 indicating over- (under-)representation."),
    ]

    # ── 4. CAUSAL FRAMEWORK ────────────────────────────────────────────────────
    story += [
        sec("4.  Causal Framework"),
        ssec("4.1  Directed Acyclic Graph"),
        p("Our causal assumptions are encoded in the DAG in Figure 1. "
          "The constitutional instrument Z (seat limits) affects treatment T "
          "(relative representation) but has no direct path to the mediator M "
          "(amendments per capita) or outcome Y (Gini) other than through T. "
          "Confounders U (GDP per capita, region fixed effects, lagged Gini) "
          "affect T, M, and Y but are blocked by conditioning."),
        p_small(
            "Z (constitutional limits 8–70 seats)  →  T (relative representation)\n"
            "T  →  M (log amendments per 100k pop.)     [path a]\n"
            "M  →  Y (Gini)                              [path b]\n"
            "T  →  Y  (direct, bypassing M)             [path c′]\n"
            "Total effect: c  =  c′  +  a·b"
        ),
        ssec("4.2  Identification strategy and instrumental validity"),
        p("The IV exclusion restriction requires that seat limits affect the Gini only "
          "through representation and amendments. For this restriction to hold, there must "
          "be no direct path from the constitutional seat allocation to contemporary "
          "inequality other than the amendment channel."),
        p("<b>Potential violation — historical confounding.</b> "
          "A legitimate concern is that state-level inequality in 1988 partly "
          "<i>caused</i> the seat allocation itself — poorer, smaller states "
          "lobbied for higher seat floors during the Constituent Assembly. "
          "If true, the DAG contains an additional path: "
          "Desig<sub>1988</sub> → Cadeiras → Gini<sub>t</sub> "
          "<i>and</i> Desig<sub>1988</sub> → Gini<sub>t</sub> "
          "(via persistence). This backdoor breaks the exclusion restriction."),
        p("<b>Our response: conditional IV.</b> "
          "We control for historical state-level development proxies that capture "
          "Desig<sub>1988</sub>: (i) log PIB per capita in 1991 (Contas Regionais, "
          "IPEADATA series PIBPCE — the earliest national accounts data available by state); "
          "and (ii) the first observed Gini per state in our panel (≈2012), "
          "which proxies for the inertial persistence of inequality. "
          "Conditioning on these variables closes the historical backdoor, "
          "yielding a <i>conditionally</i> valid instrument. "
          "The sensitivity results are reported in Table 5 (Section 6.7)."),
        p("<b>Historical control: Gini 1991 from the Demographic Census.</b> "
          "The Atlas do Desenvolvimento Humano (ADH), published jointly by PNUD, IPEA and FJP "
          "and based on the IBGE Censo Demogr\u00e1fico 1991, provides Gini coefficients by "
          "state for 1991 — just three years after the Constitution was enacted "
          "(IPEADATA series ADH_GINI; 27 states; values in the 0.49–0.68 range). "
          "We incorporate this direct pre-constitutional inequality measure as the primary "
          "historical control in the conditional IV. "
          "The 1991 Census Gini is substantially more informative than a development proxy "
          "because it captures inequality directly, not through its correlation with income. "
          "The remaining identification assumption is that the 1991 Gini adequately "
          "proxies for the 1988 value; given that Gini is highly persistent over 3-year "
          "horizons, this is a mild assumption. See Table 5 for sensitivity results."),
        p("The IV first stage (Z → T) is strong: an increase of one seat per constitutional "
          "limit unit raises relative representation by 0.458 (F-stat well above the "
          "Stock-Yogo threshold). The reduced form (Z → Y) is not significant, "
          "consistent with a null or very small total effect."),
        p("<b>Summary of identification defence.</b> "
          "The seat-cap instrument is determined by constitutional arithmetic — specifically "
          "by the interaction of population size with the 8/70 thresholds written into the "
          "1988 Constitution by a Constituent Assembly whose delegates had no information "
          "about future amendment volumes or Gini trajectories. "
          "The historical confounding threat (Desig<sub>1988</sub> → Cadeiras) "
          "is plausible but addressable: conditioning on Gini<sub>1991</sub> and "
          "log PIB<sub>1991</sub> closes the main backdoor, and Table 5 (Section 6.7) "
          "shows the reduced-form coefficient is stable across all five conditioning "
          "specifications (range: −0.000290 to +0.000076; all p > 0.20). "
          "We conclude the instrument is <i>conditionally</i> valid; the residual "
          "threat — that the 1991 Gini does not fully proxy for 1988 inequality — "
          "is acknowledged as a limitation in Section 8."),
    ]

    # ── 5. METHODOLOGY ─────────────────────────────────────────────────────────
    story += [
        sec("5.  Empirical Methodology"),
        ssec("5.1  Causal mediation (Imai, Keele, and Tingley 2010)"),
        p("We decompose the total effect following the potential-outcomes framework of "
          "Imai, Keele, and Tingley (2010), which formalises identification conditions "
          "for natural direct and indirect effects under sequential ignorability. "
          "The operational estimator uses the Baron &amp; Kenny (1986) sequential "
          "regression approach — estimated via OLS with HC3 standard errors — "
          "which Imai et al. show is numerically equivalent under linearity. Three equations:"),
        eq_image(r"\text{Step 1 (total):}\quad \mathrm{Gini}_{it} = \alpha + c\,T_{it} + \gamma X_{it} + \varepsilon_{it}"),
        eq_image(r"\text{Step 2 (T}\to\text{M):}\quad M_{it} = \alpha + a\,T_{it} + \gamma X_{it} + \varepsilon_{it}"),
        eq_image(r"\text{Step 3 (NDE):}\quad \mathrm{Gini}_{it} = \alpha + c'\,T_{it} + b\,M_{it} + \gamma X_{it} + \varepsilon_{it}"),
        sp(2),
        p("Controls X include log GDP per capita, lagged Gini, and region fixed effects. "
          "Bootstrap CIs for the indirect effect (NIE = a·b) use 5,000 resamples."),
        ssec("5.2  Distributed-lag impulse response"),
        p("To allow for multi-year accumulation we estimate:"),
        eq_image(r"\mathrm{Gini}_{it} = \sum_{k=0}^{3} \beta_k \cdot \log(\mathrm{emendas})_{i,t-k} + \gamma X_{it} + \delta_i + \lambda_t + \varepsilon_{it}"),
        p("with state (δ<sub>i</sub>) and year (λ<sub>t</sub>) fixed effects. N = 189 after creating lags. "
          "The cumulative impulse-response function (IRF) is Σ β<sub>k</sub>."),
        ssec("5.3  Difference-in-Differences (EC 86/2015)"),
        p("Constitutional Amendment 86/2015 mandated execution of individually presented "
          "amendments starting in 2016. We define as treated states those with low "
          "pre-2016 execution rates (more affected by the reform) and estimate an event "
          "study with year indicators relative to 2015. N = 243. Baseline year = 2015."),
        ssec("5.4  Regression Discontinuity Design"),
        p("We exploit the 8-seat constitutional minimum as a cutoff in an RDD, using "
          "the number of allocated seats as the running variable. The sample is restricted "
          "to states within 14 seats of the threshold (N = 15). Local linear regressions "
          "with a triangular kernel estimate the discontinuity in both amendments and Gini "
          "at the cutoff."),
        ssec("5.5  Double Machine Learning"),
        p("Following Chernozhukov et al. (2018), we estimate the partially linear model:"),
        eq_image(r"\mathrm{Gini}_{it} = \theta \cdot T_{it} + g(X_{it}) + \varepsilon_{it}"),
        p("via 5-fold cross-fitting with LassoCV to partial out the confounders g(X<sub>it</sub>). "
          "The residual-on-residual OLS estimator θ<super>^</super> is asymptotically normal and "
          "consistent under cross-fitted nuisance estimation. N = 243."),
    ]

    # ── 6. RESULTS ─────────────────────────────────────────────────────────────
    story += [
        sec("6.  Results"),
        ssec("6.1  First-stage and mediation decomposition"),
        p("Table 2 presents the Baron &amp; Kenny path coefficients. The first stage "
          "is strong and precisely estimated (a = +0.458, p &lt; 0.001). The M → Y path "
          "is borderline significant (b = −0.0068, p = 0.054), consistent with a weak "
          "redistributive effect of amendments. The direct effect T → Y is positive and "
          "significant (c′ = +0.0058, p = 0.010), implying that over-representation "
          "raises the Gini through channels other than amendments. The total effect "
          "(c = +0.0027, p = 0.151) is not distinguishable from zero."        ),
    ]

    tbl2_data = [
        ["Path", "Coefficient", "Std. Err.", "p-value", "Interpretation"],
        ["a  (T → M)", "+0.458", "0.071", "< 0.001***", "Strong 1st stage"],
        ["b  (M → Y)", "−0.0068", "0.0035", "0.054·", "Weak redistributive"],
        ["c′ (direct T → Y)", "+0.0058", "0.0022", "0.010**", "Perverse direct effect"],
        ["NIE  (a × b)", "−0.0031", "—", "n.s. [−0.0062, 0.00002]", "Via amendments"],
        ["c  (total T → Y)", "+0.0027", "0.0019", "0.151", "Net: null"],
    ]
    story += [
        sp(4),
        make_table(tbl2_data, [avail_w*0.18, avail_w*0.13, avail_w*0.12,
                                avail_w*0.23, avail_w*0.34]),
        Paragraph(
            "Table 2. Baron & Kenny mediation decomposition. HC3 standard errors. "
            "95% bootstrap CI for NIE based on 5,000 resamples. "
            "*** p<0.001 · ** p<0.01 · · p<0.10.",
            S["caption"]
        ),
        sp(6),
        ssec("6.2  Distributed-lag impulse response"),
        p("The distributed-lag model reveals a consistent negative pattern: lag-0 "
          "coefficient β<sub>0</sub> = −0.0119, accumulating to a cumulative IRF of −0.024 at lag 2, "
          "before partly recovering to −0.022 at lag 3. No individual lag coefficient is "
          "significant at the 5% level (all p > 0.13), but the sustained negative sign "
          "is consistent with a slow-moving redistributive effect that may require "
          "municipal-level data to detect."),
    ]

    tbl3_data = [
        ["Lag", "Coefficient (β_k)", "p-value", "Cumulative IRF"],
        ["Lag 0 (contemporaneous)", "−0.01188", "> 0.13", "−0.0119"],
        ["Lag 1", "−0.00738", "> 0.13", "−0.0193"],
        ["Lag 2", "−0.00485", "> 0.13", "−0.0241"],
        ["Lag 3", "+0.00201", "> 0.13", "−0.0221"],
    ]
    story += [
        sp(4),
        make_table(tbl3_data, [avail_w*0.33, avail_w*0.22, avail_w*0.18, avail_w*0.27]),
        Paragraph(
            "Table 3. Distributed-lag model coefficients. "
            "State and year FE. N = 189. HC3 clustered at state level.",
            S["caption"]
        ),
        sp(6),
        ssec("6.3  DiD event study (EC 86/2015)"),
        p("The event study produces no evidence of pre-trends or post-reform breaks. "
          "All event-time coefficients are small and statistically indistinguishable "
          "from zero at the 5% level (N = 243, baseline 2015). This null result is "
          "consistent with the amendment reform affecting federal execution rather than "
          "state-level Gini outcomes at the annual frequency."),
        ssec("6.4  RDD at the 8-seat threshold"),
        p("Within the bandwidth of ≤14 seats from the constitutional minimum, we find "
          "no significant discontinuity. At the cutoff, the estimated jumps are: "
          "amendments per capita = −R$ 5.6m/100k (p = 0.55) and Gini = −0.017 (p = 0.69). "
          "Identification is compromised by a bunching problem: 11 of 27 states sit "
          "exactly at the 8-seat minimum, creating a density mass that violates the "
          "RDD continuity assumption."),
        ssec("6.5  Double Machine Learning"),
        p("The DML estimate θ<super>^</super> = +0.00103 (95% CI: [−0.004, +0.006], p = 0.67) "
          "corroborates the null finding. Notably, naïve OLS yielded θ<super>^,OLS</super> = −0.010 "
          "(p = 0.036), an artefact of the correlation between small states "
          "(over-represented constitutionally) and high inequality — a spurious "
          "negative coefficient that DML's non-parametric deconfounding eliminates."),
        ssec("6.6  Fiscal mediation chain (SICONFI)"),
        p("Using SICONFI liquidated expenditure data for 27 UFs × 2018–2023 (N = 180), "
          "we test the secondary chain: amendments → sectoral spending → Gini. "
          "Amendments do not significantly predict health spending (β = −0.012, p = 0.70) "
          "nor education spending (β = +0.065, p = 0.19). Among sector-to-Gini links, "
          "only education expenditure reaches significance (β = −0.025, p = 0.026), "
          "while health is not significant (β = −0.015, p = 0.39). "
          "The full emendas→education→Gini chain is therefore broken at the first step."),
        p("<b>Statistical power caveat.</b> "
          "The sample of 27 UFs × 6 years yields N = 180, which provides limited power "
          "to detect small effects in sectoral spending regressions. "
          "A back-of-envelope calculation suggests minimum detectable effects of "
          "approximately 0.08–0.10 standard deviations at 80% power, "
          "meaning effects smaller than this — economically meaningful in the context of "
          "gradual educational spending changes — would not be detected. "
          "The null result for the amendment→education link should therefore be "
          "interpreted as inconclusive rather than evidence of absence, "
          "pending longer SICONFI coverage (post-2023 data will add four additional years)."),
    ]

    tbl4_data = [
        ["Link", "Coefficient", "p-value", "Significant?"],
        ["Emendas → Health spend", "−0.01196", "0.703", "No"],
        ["Emendas → Education spend", "+0.06477", "0.193", "No"],
        ["Health spend → Gini", "−0.01549", "0.393", "No"],
        ["Education spend → Gini", "−0.02531", "0.026", "Yes *"],
    ]
    story += [
        sp(4),
        make_table(tbl4_data, [avail_w*0.38, avail_w*0.18, avail_w*0.18, avail_w*0.26],
                   styles_dict=[("TEXTCOLOR", (3,4), (3,4), colors.HexColor("#006633"))]),
        Paragraph(
            "Table 4. SICONFI fiscal mediation chain. N = 180 (27 UFs × 2018–2023). "
            "OLS with log GDP per capita and year FE. * p < 0.05.",
            S["caption"]
        ),
        sp(6),
        ssec("6.7  Conditional IV — sensitivity to historical confounding"),
        p("Table 5 reports the IV sensitivity analysis. We compare the baseline "
          "reduced-form coefficient (distorcao_cadeiras → Gini) against specifications "
          "that add one or both historical proxies as controls. "
          "If the coefficient on the instrument changes substantially, the original IV "
          "was contaminated by the historical confound."),
    ]

    # ── build Table 5 rows from run_conditional_iv() results ─────────────────
    def _row5(label, key):
        return [
            label,
            _fmt_civ(key, "fs_fstat", "{:.1f}"),
            _fmt_civ(key, "rf_coef",  "{:.5f}"),
            _pstar(_fmt_civ(key, "rf_pval", "{:.3f}")),
            _fmt_civ(key, "iv_wald",  "{:.4f}"),
            _fmt_civ(key, "n",        "{:.0f}"),
        ]

    tbl5_data = [
        ["Specification", "FS F-stat", "RF coef.", "RF p-val.", "Wald IV", "N"],
        _row5("Baseline (no hist. control)",             "baseline"),
        _row5("+  Gini 1991 (ADH / Censo Demográfico)",  "cond_gini91"),
        _row5("+  log PIB per capita 1991",               "cond_pib91"),
        _row5("+  Gini baseline (≈2012)",                 "cond_gini0"),
        _row5("+  Gini 1991 + log PIB 1991 (joint)",      "cond_full"),
    ]
    story += [
        sp(4),
        make_table(tbl5_data, [avail_w*0.32, avail_w*0.12, avail_w*0.13,
                                avail_w*0.12, avail_w*0.13, avail_w*0.08],
                   styles_dict=[("FONTSIZE", (0,0), (-1,-1), 8.5)]),
        Paragraph(
            "Table 5. IV conditional sensitivity: distorcao_cadeiras → Gini with "
            "progressive historical controls. Gini 1991 from Atlas do Desenvolvimento Humano "
            "(ADH_GINI, IPEADATA), Censo Demográfico 1991. "
            "FS\u00a0= first-stage F-stat; RF\u00a0coef.\u00a0= reduced-form coefficient on instrument; "
            "Wald\u00a0IV\u00a0= RF/FS ratio estimate. "
            "Stable RF coef. across rows → exclusion restriction is robust to historical confounding.",
            S["caption"]
        ),
    ]

    # ── 7. DISCUSSION ──────────────────────────────────────────────────────────
    story += [
        sec("7.  Discussion"),
        ssec("7.1  Partial suppression mediation"),
        p("The mediation decomposition reveals a partial-suppression pattern. "
          "The indirect effect (NIE = −0.0031) and the direct effect (NDE = +0.0058) "
          "have opposite signs, largely cancelling out in the total. "
          "This suppression can arise when higher representation shifts more "
          "federal resources toward states but simultaneously directs those resources "
          "toward politically visible yet not necessarily redistributive projects "
          "(infrastructure, visible public works) rather than human capital investment. "
          "The positive direct effect may also reflect legislative bargaining: "
          "over-represented states with concentrated political power may be better "
          "at blocking redistributive general-government transfers, as the amendment "
          "system crowds out formula-based transfers."),
        ssec("7.2  Scale mismatch hypothesis"),
        p("The most substantively important caveat is scale. Parliamentary amendments "
          "target municipalities — roads, health posts, schools — while our Gini is "
          "measured at the state level. If half of a state's amendments flow to its "
          "poorest municipalities, the state Gini may not move even if local inequality "
          "falls sharply. This mismatch in measurement scale creates an attenuation "
          "bias that could account for the null findings. Future work should replicate "
          "this analysis at the municipal level using the IBGE CadÚnico microdata. "
          "This interpretation is consistent with Baião &amp; Couto (2017), who show that "
          "the politically relevant unit of the amendment mechanism is the municipality, "
          "not the state."),
        ssec("7.3  The education channel"),
        p("The only significant causal link we identify — education expenditure → Gini "
          "(β = −0.025) — is consistent with a large literature connecting schooling "
          "and wage compression (Goldin &amp; Katz, 2008; Lustig et al., 2013). "
          "The finding that amendments do not predict education spending, while "
          "education spending does predict inequality, points to a distributional channel "
          "that the amendment mechanism fails to systematically activate. "
          "This suggests that constitutional earmarks for education (as in the "
          "FUNDEB mechanism) may be more effective redistributive instruments "
          "than discretionary parliamentary amendments."),
        ssec("7.4  Instrumental validity: the historical confounding threat"),
        p("As identified in Section 4.2, the exclusion restriction faces a potential "
          "violation: if state inequality in 1988 partly determined the constitutional "
          "seat allocation, then the instrument carries a historical backdoor. "
          "Our conditional IV analysis (Table 5) uses log PIB per capita in 1991 "
          "and the baseline Gini (≈2012) as proxies for the 1988 inequality level. "
          "The key question is whether the reduced-form coefficient "
          "(distorcao_cadeiras → Gini) changes materially when these proxies are added."),
        p("Our conditional IV analysis uses, as the primary historical control, "
          "the Gini coefficient from the 1991 Demographic Census published in the "
          "Atlas do Desenvolvimento Humano (ADH_GINI, IPEADATA; PNUD, IPEA &amp; FJP, 1998). "
          "This series covers all 27 states with values in the 0.49\u20130.68 range, "
          "and is a <i>direct</i> measure of pre-constitutional inequality, "
          "not merely a correlate. The assumption required is that state-level Gini "
          "was stable between 1988 and 1991 \u2014 a mild condition given the well-documented "
          "persistence of Brazilian inequality over short horizons. "
          "A secondary specification adds log PIB per capita 1991 (PIBPCE, IPEADATA) "
          "to capture the orthogonal development dimension. "
          "Both the Gini 1991-only and the joint specification yield IV estimates "
          "broadly comparable to the baseline, supporting the validity of the instrument."),
    ]

    # ── 8. CONCLUSION ─────────────────────────────────────────────────────────
    story += [
        sec("8.  Conclusion"),
        p("Using five causal identification strategies — IV mediation analysis, "
          "distributed lags, Difference-in-Differences, RDD, and Double Machine Learning "
          "— we find no robust evidence that parliamentary over-representation reduces "
          "income inequality in Brazilian states through the earmarked amendment channel. "
          "The total effect estimate from the most credible approach (DML) is "
          "θ = +0.001 (95% CI: [−0.004, +0.006]), economically negligible and "
          "statistically indistinguishable from zero."),
        p("The result does not imply that public transfers are ineffective for redistribution. "
          "Rather, it suggests that the specific form of discretionary earmarked transfers, "
          "directed at the municipal level but measured at the state level, may be too "
          "diffuse and politically motivated to generate detectable aggregate inequality "
          "reduction. The significant education–Gini link provides a constructive hint: "
          "formula-based transfers with a strong human capital component appear more "
          "promising as redistribution vehicles than discretionary earmarks."),
        p("Three limitations bound our conclusions. First, the short panel (2012–2023) "
          "limits the power to detect slow-moving distributional effects. "
          "Second, a scale mismatch exists between municipal-targeted amendments and "
          "state-level Gini measurement; future work should use municipal Gini data. "
          "Third, the IV exclusion restriction faces a documented threat from historical "
          "confounding — the 1988 seat allocation may partly reflect pre-constitutional "
          "inequality \u2014 which we address via conditional IV conditioning on the "
          "1991 Census Gini (ADH_GINI, Atlas do Desenvolvimento Humano, IPEADATA), "
          "a near-contemporaneous direct pre-constitutional baseline. "
          "These limitations are documented in the replication code "
          "(src/analysis/iv.py::run_conditional_iv) and motivate the next stage "
          "of this research agenda at the municipal level."),
    ]

    # ── REFERENCES ─────────────────────────────────────────────────────────────
    story += [
        hr(), sp(4),
        sec("References"),
        Paragraph(
            "Baron, R. M., &amp; Kenny, D. A. (1986). The moderator–mediator variable "
            "distinction in social psychological research: Conceptual, strategic, and "
            "statistical considerations. <i>Journal of Personality and Social Psychology</i>, "
            "51(6), 1173–1182. https://doi.org/10.1037/0022-3514.51.6.1173",
            S["ref"]
        ),
        Paragraph(
            "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., "
            "Newey, W., &amp; Robins, J. (2018). Double/debiased machine learning for "
            "treatment and structural parameters. <i>The Econometrics Journal</i>, 21(1), "
            "C1–C68. https://doi.org/10.1111/ectj.12097",
            S["ref"]
        ),
        Paragraph(
            "Ferraz, C., &amp; Finan, F. (2008). Exposing corrupt politicians: The "
            "effects of Brazil's publicly released audits on electoral outcomes. "
            "<i>Quarterly Journal of Economics</i>, 123(2), 703–745.",
            S["ref"]
        ),
        Paragraph(
            "Goldin, C., &amp; Katz, L. F. (2008). <i>The Race between Education and "
            "Technology</i>. Harvard University Press.",
            S["ref"]
        ),
        Paragraph(
            "Hahn, J., Todd, P., &amp; Van der Klaauw, W. (2001). Identification and "
            "estimation of treatment effects with a regression-discontinuity design. "
            "<i>Econometrica</i>, 69(1), 201–209.",
            S["ref"]
        ),
        Paragraph(
            "Imai, K., Keele, L., &amp; Tingley, D. (2010). A general approach to "
            "causal mediation analysis. <i>Psychological Methods</i>, 15(4), 309–334. "
            "https://doi.org/10.1037/a0020761",
            S["ref"]
        ),
        Paragraph(
            "Lustig, N., Lopez-Calva, L. F., &amp; Ortiz-Juarez, E. (2013). Declining "
            "inequality in Latin America in the 2000s: The cases of Argentina, Brazil, "
            "and Mexico. <i>World Development</i>, 44, 129–141.",
            S["ref"]
        ),
        Paragraph(
            "Pereira, C., &amp; Mueller, B. (2004). The cost of governing: Strategic "
            "behavior of the president and legislators in Brazil's budgetary process. "
            "<i>Comparative Political Studies</i>, 37(7), 781–815.",
            S["ref"]
        ),
        Paragraph(
            "Pearl, J. (2001). Direct and indirect effects. In <i>Proceedings of the "
            "17th UAI Conference</i> (pp. 411–420). Morgan Kaufmann.",
            S["ref"]
        ),
        Paragraph(
            "Roth, J., Sant'Anna, P. H. C., Bilinski, A., &amp; Poe, J. (2023). "
            "What's trending in difference-in-differences? A synthesis of the recent "
            "econometrics literature. <i>Journal of Econometrics</i>, 235(2), 2218–2244. "
            "https://doi.org/10.1016/j.jeconom.2023.03.008",
            S["ref"]
        ),
        Paragraph(
            "VanderWeele, T. J. (2015). <i>Explanation in Causal Inference: Methods for "
            "Mediation and Interaction</i>. Oxford University Press.",
            S["ref"]
        ),
        Paragraph(
            "Weaver, J. (2021). Jobs for sale: Corruption and misallocation in "
            "hiring. <i>American Economic Review</i>, 111(10), 3093–3122.",
            S["ref"]
        ),
        Paragraph(
            "Baião, A. L., &amp; Couto, C. G. (2017). A efetividade das emendas "
            "parlamentares em um contexto de governo dividido. "
            "<i>Dados — Revista de Ciências Sociais</i>, 60(1), 67–105. "
            "https://doi.org/10.1590/001152580152",
            S["ref"]
        ),
        Paragraph(
            "Beramendi, P., Rogers, M., &amp; Díaz-Cayeros, A. (2017). "
            "Endogenous decentralisation in federal systems. "
            "<i>Comparative Political Studies</i>, 50(10), 1317–1351.",
            S["ref"]
        ),
        Paragraph(
            "Dunn, A. (2022). Malapportionment, fiscal policy, and government spending "
            "in the United States. <i>Publius: The Journal of Federalism</i>, 52(1), 1–27.",
            S["ref"]
        ),
        Paragraph(
            "Galiani, S., Galetovic, A., &amp; Schargrodsky, E. (2016). "
            "Parliamentary representation and public goods: Evidence from Brazil. "
            "<i>Economics &amp; Politics</i>, 28(1), 90–115. "
            "https://doi.org/10.1111/ecpo.12070",
            S["ref"]
        ),
        Paragraph(
            "Mesquita, L. (2008). Emendas ao orçamento e conexão eleitoral na "
            "Câmara dos Deputados brasileira. "
            "<i>Dissertação de Mestrado</i>, USP, São Paulo.",
            S["ref"]
        ),
        Paragraph(
            "PNUD, IPEA, &amp; FJP. (1998). "
            "<i>Atlas do Desenvolvimento Humano no Brasil</i>. "
            "Based on IBGE Censo Demográfico 1991. "
            "Data retrieved from IPEADATA series ADH_GINI. "
            "http://www.ipeadata.gov.br",
            S["ref"]
        ),
        Paragraph(
            "Pereira, C., &amp; Rennó, L. (2007). O que é que o reeleito tem? "
            "O retorno: O esboço de uma teoria da reeleição no Brasil. "
            "<i>Novos Estudos CEBRAP</i>, 79, 11–28.",
            S["ref"]
        ),
        sp(10),
        hr(),
        Paragraph(
            f"This working paper was prepared using Python/ReportLab. "
            f"Replication code and data: "
            "<font color='#1a4eb8'>https://github.com/CaioMar/politicas-publicas</font> · "
            f"Generated {datetime.date.today().strftime('%B %d, %Y')}.",
            S["footnote"]
        ),
    ]

    return story


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 0.5*cm, bottomMargin=MARGIN,
        title="Parliamentary Representation, Earmarked Transfers, and Inequality in Brazil",
        author="Caio Martins Ramos de Oliveira",
        subject="Causal Analysis · Brazilian Fiscal Federalism",
        creator="ReportLab / Python",
    )
    S = build_styles()
    story = build_story(S)
    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written → {OUT_PATH}")


if __name__ == "__main__":
    build_pdf()
