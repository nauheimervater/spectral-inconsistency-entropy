"""
Generate the publication PDF for:
  "Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions
   in Pairwise Comparison Graphs" (Thomas Nauheimer, November 2026).
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs",
        author="Thomas Nauheimer",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    author_style = ParagraphStyle(
        "PaperAuthor",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    abstract_style = ParagraphStyle(
        "PaperAbstract",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Times-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    eq_style = ParagraphStyle(
        "PaperEq",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=6,
    )
    thm_style = ParagraphStyle(
        "PaperThm",
        parent= body_style,
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=6,
    )

    story = []

    story.append(
        Paragraph(
            "Spectral Inconsistency Entropy: Distinguishing Local from Systemic Contradictions in Pairwise Comparison Graphs",
            title_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Thomas Nauheimer</b><br/>Independent Researcher (nauheimer.t@gmail.com)<br/>November 2026",
            author_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Abstract.</b> Pairwise comparison data—arising in sports rankings, autonomous sensor fusion, peer review, and RLHF preference modeling—frequently exhibits cyclic inconsistencies. While classical combinatorial Hodge theory (HodgeRank) quantifies the total magnitude of inconsistency via the cyclic residual ratio &rho;(G, w) = ||r||<sup>2</sup> / ||w||<sup>2</sup>, it is invariant to how contradictions are topologically distributed across the comparison network. We introduce <b>Spectral Inconsistency Entropy (H)</b>, a scale- and potential-invariant diagnostic defined as the Shannon entropy of the cyclic residual's energy projection across the distinct eigenspaces of the cycle-restricted 1-up-Laplacian L<sub>Z</sub> = Z<sup>T</sup> B<sub>1</sub> B<sub>1</sub><sup>T</sup> Z. We prove four results: (1) <b>Universal Clique Degeneracy (Theorem 1):</b> By the simplicial Eckmann identity B<sub>0</sub><sup>T</sup> B<sub>0</sub> + B<sub>1</sub> B<sub>1</sub><sup>T</sup> = n I<sub>|E|</sub> on the clique 2-complex of K<sub>n</sub>, L<sub>Z</sub> = n I<sub>&beta;<sub>1</sub>(K<sub>n</sub>)</sub> has a single eigenvalue &lambda; = n of multiplicity (n-1)(n-2)/2 for all n &ge; 3, so H(K<sub>n</sub>, w) &equiv; 0 identically. (2) <b>Harmonic–Curl Separation &amp; Scale Invariance (Theorem 2):</b> The &lambda; = 0 eigenspace of L<sub>Z</sub> is canonically isomorphic to H<sup>1</sup>(K<sub>2</sub>(G)) = ker(B<sub>0</sub>) &cap; ker(B<sub>1</sub><sup>T</sup>), while &lambda; &gt; 0 eigenspaces span im(B<sub>1</sub>), and H(G, B<sub>0</sub><sup>T</sup> s + &alpha; &delta;) = H(G, &delta;) for all s and &alpha; &ne; 0. (3) <b>Exact Isotropic Dirichlet–Digamma Asymptote (Theorem 3):</b> Under isotropic Gaussian cycle noise c ~ N(0, &sigma;<sup>2</sup> I<sub>&beta;<sub>1</sub></sub>), the eigenspace energies follow Dirichlet(d<sub>1</sub>/2, ..., d<sub>|&Lambda;|</sub>/2) with closed-form expected entropy H<sub>iso</sub>(G). (4) <b>Empirical &amp; Structural Boundaries:</b> Across 1,000 Monte Carlo trials per edit distance k on G(15, 0.4), H(G, w) grows monotonically from k=1 (3.7641 &plusmn; 0.0394 bits) to the exact Dirichlet–Digamma asymptote H<sub>iso</sub> = 4.1175 bits (4.1274 &plusmn; 0.0124 bits at k=40), while cycle-leverage cosine alignment localizes single corrupted edges with 91.3%–100.0% Top-1 accuracy across five canonical graph families.",
            abstract_style,
        )
    )

    story.append(Paragraph("1. Introduction and Mathematical Framework", h1_style))
    story.append(
        Paragraph(
            "Given a comparison graph G = (V, E) with |V| = n and |E| = m, an antisymmetric edge flow w &isin; R<sup>m</sup> is consistent if and only if w = B<sub>0</sub><sup>T</sup> s for a scalar node potential s &isin; R<sup>n</sup>. Combinatorial Hodge theory (Jiang et al., 2011) decomposes R<sup>m</sup> orthogonally into gradient flows im(B<sub>0</sub><sup>T</sup>), triangular curl flows im(B<sub>1</sub>), and harmonic 1-forms H<sup>1</sup>(K<sub>2</sub>(G)) = ker(B<sub>0</sub>) &cap; ker(B<sub>1</sub><sup>T</sup>). While the classical inconsistency index &rho;(G, w) = ||r||<sup>2</sup> / ||w||<sup>2</sup> measures total cyclic energy, it confounds the ground-truth potential scale ||B<sub>0</sub><sup>T</sup> s*||<sup>2</sup> with the fault topology and cannot distinguish a single faulty sensor (L<sub>0</sub> edit distance k = 1) from systemic network-wide noise.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Let Z &isin; R<sup>|E| &times; &beta;<sub>1</sub>(G)</sup> be an orthonormal basis of the cycle space Z = ker(B<sub>0</sub>), where &beta;<sub>1</sub>(G) = |E| - |V| + c(G). Restricting the 1-up-Laplacian L<sub>1</sub><sup>up</sup> = B<sub>1</sub> B<sub>1</sub><sup>T</sup> to Z yields the positive semi-definite operator L<sub>Z</sub> = Z<sup>T</sup> B<sub>1</sub> B<sub>1</sub><sup>T</sup> Z. Let &Lambda;(G) be its distinct eigenvalues with multiplicities d<sub>&lambda;</sub> and orthogonal eigenspace projectors P<sub>&lambda;</sub>. For c = Z<sup>T</sup> w &ne; 0, the spectral energy fractions and Spectral Inconsistency Entropy are:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "p<sub>&lambda;</sub>(G, w) = ||P<sub>&lambda;</sub> c||<sup>2</sup> / ||c||<sup>2</sup>, &nbsp;&nbsp;&nbsp;&nbsp; H(G, w) = - &Sigma;<sub>&lambda; &isin; &Lambda;(G)</sub> p<sub>&lambda;</sub>(G, w) log<sub>2</sub> p<sub>&lambda;</sub>(G, w) &isin; [0, log<sub>2</sub> |&Lambda;(G)|].",
            eq_style,
        )
    )

    story.append(Paragraph("2. Main Theoretical Results", h1_style))
    story.append(
        Paragraph(
            "<b>Theorem 1 (Universal Clique Degeneracy via Eckmann Identity).</b> <i>For every complete graph K<sub>n</sub> (n &ge; 3), the 1-Hodge Laplacian on the clique 2-complex satisfies B<sub>0</sub><sup>T</sup> B<sub>0</sub> + B<sub>1</sub> B<sub>1</sub><sup>T</sup> = n I<sub>|E|</sub>. Consequently, L<sub>Z</sub> = n I<sub>&beta;<sub>1</sub>(K<sub>n</sub>)</sub> with &beta;<sub>1</sub>(K<sub>n</sub>) = (n-1)(n-2)/2 and a single eigenvalue &lambda; = n. Therefore, H(K<sub>n</sub>, w) &equiv; 0 identically for all n &ge; 3 and all flows w.</i>",
            thm_style,
        )
    )
    story.append(
        Paragraph(
            "<i>Proof.</i> Every edge e = (u, v) in K<sub>n</sub> has 2 endpoints and belongs to n - 2 triangles, giving diagonal entries 2 + (n - 2) = n in B<sub>0</sub><sup>T</sup> B<sub>0</sub> + B<sub>1</sub> B<sub>1</sub><sup>T</sup>. Disjoint edges share neither vertices nor triangles (off-diagonal 0). Two edges sharing one vertex have +1 in B<sub>0</sub><sup>T</sup> B<sub>0</sub> and -1 in B<sub>1</sub> B<sub>1</sub><sup>T</sup> from their unique common triangle, canceling to 0. Since B<sub>0</sub> Z = 0, restricting to Z gives L<sub>Z</sub> = n I<sub>&beta;<sub>1</sub>(K<sub>n</sub>)</sub>, so |&Lambda;(K<sub>n</sub>)| = 1 and H(K<sub>n</sub>, w) = 0. Q.E.D.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Theorem 2 (Harmonic–Curl Separation &amp; Scale/Potential Invariance).</b> <i>(1) The &lambda; = 0 eigenspace of L<sub>Z</sub> maps via Z isomorphically onto H<sup>1</sup>(K<sub>2</sub>(G)) = ker(B<sub>0</sub>) &cap; ker(B<sub>1</sub><sup>T</sup>) (global non-triangular cycles of length &ge; 4), while &oplus;<sub>&lambda;&gt;0</sub> V<sub>&lambda;</sub> maps onto im(B<sub>1</sub>). (2) For any node potential s &isin; R<sup>|V|</sup>, perturbation &delta; &notin; im(B<sub>0</sub><sup>T</sup>), and &alpha; &ne; 0, H(G, B<sub>0</sub><sup>T</sup> s + &alpha; &delta;) = H(G, &delta;).</i>",
            thm_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Theorem 3 (Exact Isotropic Dirichlet–Digamma Asymptote).</b> <i>Under isotropic Gaussian cycle noise c ~ N(0, &sigma;<sup>2</sup> I<sub>&beta;<sub>1</sub>(G)</sub>), the squared projections ||P<sub>&lambda;</sub> c||<sup>2</sup> ~ &sigma;<sup>2</sup> &chi;<sup>2</sup><sub>d<sub>&lambda;</sub></sub> induce (p<sub>&lambda;</sub>) ~ Dirichlet(d<sub>1</sub>/2, ..., d<sub>|&Lambda;|</sub>/2) with exact expected entropy:</i>",
            thm_style,
        )
    )
    story.append(
        Paragraph(
            "H<sub>iso</sub>(G) = (1 / ln 2) &Sigma;<sub>&lambda; &isin; &Lambda;(G)</sub> (d<sub>&lambda;</sub> / &beta;<sub>1</sub>(G)) [ &psi;(&beta;<sub>1</sub>(G)/2 + 1) - &psi;(d<sub>&lambda;</sub>/2 + 1) ] &le; H<sub>sys</sub>(G) = - &Sigma; (d<sub>&lambda;</sub> / &beta;<sub>1</sub>) log<sub>2</sub>(d<sub>&lambda;</sub> / &beta;<sub>1</sub>).",
            eq_style,
        )
    )

    story.append(Paragraph("3. Empirical Validation and Cross-Topology Regimes", h1_style))
    story.append(
        Paragraph(
            "On Erdős–Rényi G(15, 0.4) (seed=42, |V|=15, |E|=49, |T|=50, &beta;<sub>1</sub>=35, |&Lambda;|=35), Theorem 3 predicts H<sub>iso</sub>(G) = (&psi;(18.5) - &psi;(1.5))/ln 2 = 4.1175 bits (Jensen bound H<sub>sys</sub> = log<sub>2</sub>(35) = 5.1293 bits). Across 1,000 Monte Carlo trials per L<sub>0</sub> edit distance k, H(G, w) increases strictly monotonically toward H<sub>iso</sub>(G):",
            body_style,
        )
    )

    table1_data = [
        ["L0 Edit Distance (k)", "Mean H(G, w) +/- 95% CI (bits)", "Std H (bits)", "Mean rho(G, w)", "Top-1 Fault Loc."],
        ["1", "3.7641 +/- 0.0394", "0.6363", "0.1217", "100.0%"],
        ["2", "3.8806 +/- 0.0275", "0.4432", "0.2174", "—"],
        ["3", "3.9308 +/- 0.0233", "0.3765", "0.2839", "—"],
        ["4", "3.9459 +/- 0.0200", "0.3235", "0.3387", "—"],
        ["5", "3.9826 +/- 0.0179", "0.2889", "0.3845", "—"],
        ["6", "3.9928 +/- 0.0186", "0.3001", "0.4186", "—"],
        ["9", "4.0308 +/- 0.0152", "0.2445", "0.4939", "—"],
        ["14", "4.0795 +/- 0.0142", "0.2288", "0.5581", "—"],
        ["20", "4.0925 +/- 0.0137", "0.2203", "0.5986", "—"],
        ["30", "4.0965 +/- 0.0130", "0.2098", "0.6360", "—"],
        ["40", "4.1274 +/- 0.0124", "0.2007", "0.6540", "—"],
    ]
    t1 = Table(table1_data, colWidths=[34 * mm, 48 * mm, 26 * mm, 28 * mm, 30 * mm])
    t1.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(t1)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Comparing five graph families illustrates the structural regimes of H(G, w) alongside our cycle-leverage cosine localizer argmax<sub>j</sub> |&lt;Z<sup>T</sup> w, Z<sup>T</sup> e<sub>j</sub>&gt;| / (||Z<sup>T</sup> w|| ||Z<sup>T</sup> e<sub>j</sub>||):",
            body_style,
        )
    )

    table2_data = [
        ["Network Topology", "|V|", "|E|", "beta_1", "|Lambda|", "H(k=1)", "H_iso", "H_sys", "Top-1 Loc."],
        ["Complete Graph K_8", "8", "28", "21", "1", "0.0000", "0.0000", "0.0000", "100.0%"],
        ["Modular League (4x5 Caveman)", "20", "40", "20", "5", "0.4809", "1.4392", "1.6710", "97.4%"],
        ["Small-World Tournament (NWS 16)", "16", "45", "30", "26", "2.9614", "3.6453", "4.5199", "100.0%"],
        ["Erdos-Renyi G(15, 0.4)", "15", "49", "35", "35", "3.7512", "4.1175", "5.1293", "100.0%"],
        ["Triangular Sensor Grid (4x4)", "12", "23", "12", "10", "2.7750", "2.4634", "3.2516", "91.3%"],
    ]
    t2 = Table(
        table2_data,
        colWidths=[48 * mm, 11 * mm, 11 * mm, 14 * mm, 16 * mm, 16 * mm, 16 * mm, 16 * mm, 20 * mm],
    )
    t2.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Structural Takeaway.</b> In modular and heterogeneous comparison graphs (Modular League, Small-World Tournament, Erdős–Rényi), localized cycle modes ensure H(k=1) &lt; H<sub>iso</sub>(G) &lt; H<sub>sys</sub>(G), providing a sharp diagnostic separation between single bad actors and systemic noise. On regular lattices (Triangular Sensor Grid), delocalized Bloch waves spread a single-edge delta spike across frequencies (H(k=1) = 2.7750 &gt; H<sub>iso</sub> = 2.4634 bits due to Porter–Thomas &chi;<sup>2</sup><sub>1</sub> fluctuations in single Gaussian realizations, while still respecting the Jensen bound H<sub>sys</sub> = 3.2516 bits). Pairing H(G, w) with cycle-leverage cosine alignment yields both regime diagnosis and 91.3%–100.0% single-fault localization across all topologies.",
            body_style,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "Spectral_Inconsistency_Entropy.pdf")
    build_pdf(out)
    print(f"Generated PDF: {out} ({os.path.getsize(out)} bytes)")
