const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle,
} = require("docx");

const dir = __dirname;
const FONT = "Arial";
const TEAL = "064A56";

const p = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120, line: 276 }, ...opts, children: runs(text) });

// Simple **bold** markup support.
function runs(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true }) : new TextRun(s)
  );
}
const bullet = (text) =>
  new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(text) });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const caption = (t) =>
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });
const img = (file, w, h) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(dir, file)), transformation: { width: w, height: h } })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D3D5" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, boldRow = -1) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, isHead, bold) =>
    new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: isHead ? { fill: TEAL, type: ShadingType.CLEAR, color: "auto" } : bold ? { fill: "E8F1F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ alignment: i > 0 && /^[\d$\u2248\u2013]/.test(text) ? AlignmentType.RIGHT : AlignmentType.LEFT,
        children: [new TextRun({ text, bold: isHead || bold, color: isHead ? "FFFFFF" : "222222", size: 19 })] })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, ri === boldRow)) })),
    ],
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 40, bold: true, font: FONT, color: TEAL }, paragraph: { spacing: { after: 80 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: TEAL }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "222222" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1200, bottom: 1200, left: 1300, right: 1300 } } },
    children: [
      new Paragraph({ style: "Title", children: [new TextRun("Freight Rate Prediction")] }),
      new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 6 } },
        children: [new TextRun({ text: "Spotter Machine Learning Engineer assessment — validation approach and results", color: "555555" })] }),

      h1("Summary"),
      p("The labelled data covers January–October 2025 and the 12,000 loads to predict fall in November–December 2025, with no overlapping dates. I therefore treated the task as a **two-month-ahead forecast** and validated only on time-based splits that mimic it. On the most realistic holdout (train January–August, test September–October) the final model reaches **1.66% MAPE ($39 MAE)**, against 3.74% for a lane-median baseline and 4.96% for a standard gradient-boosting model."),

      h1("1. Splitting and validating the data"),
      h2("Why not a random split"),
      p("A random 80/20 split lets the model see loads from the same days it is tested on, which hides two problems that matter for November–December: (1) prices drift over time, and (2) one input column, quote_signal, leaks the answer on some days. On a random split a standard gradient-boosting model scores 1.66% MAPE and adding quote_signal “improves” it to 1.52%. On a time split the same models score 4–5%. The random split would have rewarded exactly the wrong choices."),
      h2("Time-based folds"),
      p("Each fold trains on earlier months and tests on the following two months, matching the real gap between training and validation data:"),
      bullet("**Fold A:** train January–June, test July–August."),
      bullet("**Fold B:** train January–August, test September–October. This is the closest analogue to the real task: it contains a quarter-end in the test window, as December does."),
      p("All cleaning rules and reference tables are built without using the target, and the corrupted-label filter is applied only to training rows. Metrics are reported on clean test loads, with MAE on all rows shown for completeness. After model selection the final model is refit on all clean January–October loads."),

      h1("2. What the data showed"),
      img("eda_findings.png", 600, 193),
      caption("Left: quote_signal equals the true rate-per-mile on most loads in some months and almost none in others. Right: the daily price level follows the market index plus an upward trend, with spikes at each quarter-end."),
      bullet("**quote_signal is unreliable and was dropped.** On about half the training days it is within 2% of the true rate-per-mile (a leak); on the other days it is weakly negatively related. In November–December it matches neither pattern: it has zero correlation with distance and no equipment difference, so it carries no information where predictions are needed."),
      bullet("**market_index is a daily market level.** Its daily mean tracks the daily price level (correlation 0.70); within a day it is mostly noise. The model uses the daily mean, which also provides a market level for the December chart rows."),
      bullet("**Prices drift upward about 7.6% per year** beyond what the market index explains, and **rise by about 4% in the last days of each quarter**, ramping up over the final two weeks."),
      h2("Data-quality issues and fixes"),
      table(["Issue", "Rows (train / validation)", "Fix"], [
        ["Corrupted rates (0.2–0.4× or 2.5–4.5× expected)", "677 / –", "Excluded from training"],
        ["Negative weights (sign errors)", "292 / 145", "Absolute value"],
        ["Missing weight", "300 / 165", "Equipment median + flag"],
        ["Missing market_index", "374 / 249", "Daily mean index"],
        ["quote_signal leak", "≈ 50% of days", "Column dropped"],
      ], [3900, 2200, 3240]),

      h1("3. Model"),
      p("The model has two stages, both on log(rate) so errors are proportional to price:"),
      bullet("**Linear backbone (ridge regression):** pickup, delivery and equipment effects, log distance with a slope per equipment type, weight, log daily market index, a linear time trend, quarter-end and quarter-start indicators, and day of week. These effects must extrapolate two months beyond the training dates, which linear terms can do and tree models cannot."),
      bullet("**Gradient-boosted trees on the backbone’s residuals:** capture non-linear structure such as distance curvature and lane-specific quirks, using only features that do not depend on the calendar date."),

      h1("4. Results"),
      table(["Model", "Fold A MAPE", "Fold B MAPE", "Fold B MAE"], [
        ["Lane median $/mile", "4.28%", "3.74%", "$88"],
        ["Ridge (no trend / quarter-end)", "2.30%", "4.43%", "$108"],
        ["Gradient boosting", "2.43%", "4.96%", "$120"],
        ["Gradient boosting + quote_signal", "2.76%", "4.16%", "$99"],
        ["Two-stage model (final)", "3.03%", "1.66%", "$39"],
      ], [3900, 1800, 1800, 1840], 4),
      p(""),
      p("Fold A is weaker for the final model because in January–June the market index rises almost in step with time (correlation 0.86), so the trend cannot be separated from the index. Once the index cycles down in the summer the two are separable; across the full January–October data their correlation is −0.11, and the fitted effects are stable (index elasticity 0.16, trend +7.6%/year, quarter-end uplift +4.0%)."),

      h1("5. December prediction chart"),
      img("candidate_december.png", 600, 194),
      caption("Produced by the provided score.py. Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes."),
      p("The forecast rises from about $830 to $878 over the month. The weekly wave follows the market index, which peaks on Thursdays in the validation data. The upward path combines the long-run trend with the quarter-end ramp, reaching the month’s high on 31 December. These levels are consistent with recent Dry Van loads on this lane ($780–$865 in August–October)."),

      h1("6. Limitations"),
      bullet("There is no November or December history, so holiday effects such as Christmas are not learned beyond the general quarter-end pattern."),
      bullet("The trend is assumed linear and is extrapolated two months ahead."),
      bullet("Corrupted rates cannot be predicted; if the final metric includes such rows, every model’s error will be higher."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(dir, "freight_rate_report.docx"), buf);
  console.log("wrote report");
});
