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
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, datetime

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
                      rightIndent=1.2*cm, spaceAfter=10, alignment=TA_JUSTIFY,
                      textColor=gray),
        "keywords": S("PKW", fontSize=8.5, leading=12, leftIndent=1.2*cm,
                      rightIndent=1.2*cm, spaceAfter=16, textColor=gray,
                      fontName="Helvetica-Oblique"),
        "section":  S("PSec", fontSize=12, leading=15, spaceBefore=18,
                      spaceAfter=6, fontName="Helvetica-Bold", textColor=dark),
        "subsection": S("PSSec", fontSize=10.5, leading=13, spaceBefore=10,
                        spaceAfter=4, fontName="Helvetica-Bold", textColor=dark),
        "body":     S("PBody", fontSize=10, leading=14.5, spaceAfter=7,
                      alignment=TA_JUSTIFY),
        "body_small": S("PSmall", fontSize=9, leading=13, spaceAfter=6,
                        alignment=TA_JUSTIFY, textColor=gray),
        "caption":  S("PCap", fontSize=8.5, leading=12, spaceAfter=8,
                      fontName="Helvetica-Oblique", textColor=gray, alignment=TA_CENTER),
        "footnote": S("PFn", fontSize=8, leading=11, spaceAfter=3,
                      textColor=gray),
        "equation": S("PEq", fontSize=9.5, leading=14, leftIndent=1.5*cm,
                      spaceAfter=5, fontName="Courier"),
        "ref":      S("PRef", fontSize=8.5, leading=12, leftIndent=0.6*cm,
                      firstLineIndent=-0.6*cm, spaceAfter=5, textColor=gray),
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
        ("FONTNAME",    (0,0), (-1, hdr_rows-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1, hdr_rows-1), 9),
        ("FONTNAME",    (0, hdr_rows), (-1,-1), "Helvetica"),
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
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    # Running header
    header = "Martins, C. — Parliamentary Representation, Earmarked Transfers & Inequality"
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
        Paragraph("Caio Martins", S["author"]),
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
            "strategies — mediation analysis (Baron &amp; Kenny), distributed-lag impulse "
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
          "framework."),
    ]

    # ── 2. INSTITUTIONAL BACKGROUND ────────────────────────────────────────────
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
        ssec("4.2  Identification strategy"),
        p("The IV exclusion restriction requires that seat limits affect the Gini only "
          "through representation and amendments. We argue this is plausible: the "
          "constitutional minimum was set in 1988 based on political bargaining "
          "unrelated to contemporary inequality trajectories, and there is no direct "
          "federal channel from seat count to state inequality outside the amendment mechanism."),
        p("The IV first stage (Z → T) is strong: an increase of one seat per constitutional "
          "limit unit raises relative representation by 0.458 (F-stat well above the "
          "Stock-Yogo threshold). The reduced form (Z → Y) is not significant, "
          "consistent with a null or very small total effect."),
    ]

    # ── 5. METHODOLOGY ─────────────────────────────────────────────────────────
    story += [
        sec("5.  Empirical Methodology"),
        ssec("5.1  Causal mediation (Baron & Kenny / Baron & Kenny 1986)"),
        p("We decompose the total effect following the Baron &amp; Kenny (1986) sequential "
          "regression approach, estimated via OLS with heteroskedasticity-robust standard "
          "errors (HC3). Three equations:"),
        Paragraph("Step 1 (total effect):  Gini_{it} = α + c·T_{it} + γX_{it} + ε_{it}", Eq),
        Paragraph("Step 2 (T → M):         M_{it} = α + a·T_{it} + γX_{it} + ε_{it}", Eq),
        Paragraph("Step 3 (NDE, M → Y):    Gini_{it} = α + c′·T_{it} + b·M_{it} + γX_{it} + ε_{it}", Eq),
        p("Controls X include log GDP per capita, lagged Gini, and region fixed effects. "
          "Bootstrap CIs for the indirect effect (NIE = a·b) use 5,000 resamples."),
        ssec("5.2  Distributed-lag impulse response"),
        p("To allow for multi-year accumulation we estimate:"),
        Paragraph(
            "Gini_{it} = Σ_{k=0}^{3} β_k · log(emendas)_{i,t-k} + γX_{it} + δ_i + λ_t + ε_{it}",
            Eq
        ),
        p("with state (δ_i) and year (λ_t) fixed effects. N = 189 after creating lags. "
          "The cumulative impulse-response function (IRF) is Σ β_k."),
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
        Paragraph("Gini_{it} = θ · T_{it} + g(X_{it}) + ε_{it}", Eq),
        p("via 5-fold cross-fitting with LassoCV to partial out the confounders g(X). "
          "The residual-on-residual OLS estimator θ̂ is asymptotically normal and "
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
          "(c = +0.0027, p = 0.151) is not distinguishable from zero."),
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
          "coefficient β₀ = −0.0119, accumulating to a cumulative IRF of −0.024 at lag 2, "
          "before partly recovering to −0.022 at lag 3. No individual lag coefficient is "
          "significant at the 5% level (all p > 0.13), but the sustained negative sign "
          "is consistent with a slow-moving redistributive effect that may require "
          "municipal-level data to detect."),
    ]

    tbl3_data = [
        ["Lag", "Coefficient (βₖ)", "p-value", "Cumulative IRF"],
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
        p("The DML estimate θ̂ = +0.00103 (95% CI: [−0.004, +0.006], p = 0.67) "
          "corroborates the null finding. Notably, naïve OLS yielded θ̂ᴼᴸˢ = −0.010 "
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
          "this analysis at the municipal level using the IBGE CadÚnico microdata."),
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
        p("Our analysis is constrained by the short panel (2012–2023), the potential "
          "scale mismatch between amendment targeting and Gini measurement, and the "
          "bunching issue at the RDD threshold. Future research should exploit "
          "municipal-level Gini estimates and heterogeneous treatment effect analysis "
          "to uncover distributional effects that aggregate Gini metrics may mask."),
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
        author="Caio Martins",
        subject="Causal Analysis · Brazilian Fiscal Federalism",
        creator="ReportLab / Python",
    )
    S = build_styles()
    story = build_story(S)
    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    print(f"PDF written → {OUT_PATH}")


if __name__ == "__main__":
    build_pdf()
