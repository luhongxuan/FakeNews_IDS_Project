import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const bridgeRoot = path.join(repoRoot, "benchmarks", "pheme_static_dynamic_bridge");
const separateRun = path.join(
  repoRoot, "analysis", "benchmarks", "pheme_static_dynamic_bridge", "20260907_173636_206893_separate_fixed30_pi30_and_dynamic_pi_t",
);
const bridgeRun = path.join(
  bridgeRoot, "reference_result", "20260907_154306_938376_bridge_a_full",
);
const inputPaths = {
  fixed: path.join(separateRun, "fixed30_pi30_over_pi30.csv"),
  dynamic: path.join(separateRun, "dynamic_pi_selected_over_pi10_macro.csv"),
  dynamicCheckpoint: path.join(separateRun, "dynamic_pi_t_over_pi_t_per_checkpoint.csv"),
  bridge: path.join(bridgeRun, "unified_bridge_a_model_comparison.csv"),
  bridgePerEvent: path.join(bridgeRun, "per_event_bridge_a_metrics.csv"),
};
const primaryEvents = new Set([
  "charliehebdo", "ferguson", "germanwings-crash", "ottawashooting",
  "prince-toronto", "putinmissing", "sydneysiege",
]);

function timestamp() {
  const d = new Date();
  const pad = (v, n = 2) => String(v).padStart(n, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}_${pad(d.getMilliseconds(), 3)}`;
}

function isoNow() { return new Date().toISOString(); }

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows.filter((r) => r.some((v) => v !== ""));
}

const numericPattern = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
function typedRows(rows) {
  return rows.map((row, rowIndex) => row.map((value) => {
    if (rowIndex > 0 && numericPattern.test(value)) return Number(value);
    return value;
  }));
}

function colName(n) {
  let result = "", x = n;
  while (x > 0) { x -= 1; result = String.fromCharCode(65 + (x % 26)) + result; x = Math.floor(x / 26); }
  return result;
}

function styleDataSheet(sheet, rows, tableName) {
  const rowCount = rows.length;
  const colCount = rows[0].length;
  const lastCol = colName(colCount);
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastCol}${rowCount}`).values = rows;
  sheet.getRange(`A1:${lastCol}${rowCount}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
  sheet.getRange(`A1:${lastCol}1`).format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A1:${lastCol}${rowCount}`).format.verticalAlignment = "center";
  sheet.getRange(`A1:${lastCol}${rowCount}`).format.autofitColumns();
  sheet.getRange(`A1:${lastCol}${rowCount}`).format.autofitRows();
  sheet.getRange(`A1:${lastCol}${rowCount}`).format.columnWidth = 18;
  sheet.getRange("A:A").format.columnWidth = 31;
  sheet.getRange(`A1:${lastCol}1`).format.rowHeight = 42;
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(`A1:${lastCol}${rowCount}`, true, tableName);
  table.style = "TableStyleMedium2";
  for (let c = 0; c < rows[0].length; c += 1) {
    const header = String(rows[0][c]).toLowerCase();
    if (header.includes("score") || header.includes("capture") || header.includes("recall")) {
      sheet.getRange(`${colName(c + 1)}2:${colName(c + 1)}${rowCount}`).format.numberFormat = "0.00%";
    }
  }
}

async function main() {
  const outputDir = path.join(bridgeRoot, "experiments", `${timestamp()}_three_stage_paper_results_bundle`);
  await fs.mkdir(outputDir, { recursive: false });
  const runRecordPath = path.join(outputDir, "run_record.json");
  const runRecord = {
    status: "running",
    started_at: isoNow(),
    configuration: {
      purpose: "paper-ready three-stage fixed, dynamic, and Bridge result consolidation",
      primary_aggregation: "pooled; macro retained as event-balanced robustness metric",
      training_performed: false,
    },
    input_artifacts: inputPaths,
    split: "same 2,373-thread contract; 7-event primary reporting",
    cutoff: "fixed=30min; dynamic=10/20/30/40/50/60min; Bridge starts at 10min",
    seed: 42,
    metrics_results: {},
    output_file_inventory: [],
    failure_details: null,
  };
  await fs.writeFile(runRecordPath, JSON.stringify(runRecord, null, 2));
  console.log("Starting three-stage research result workbook build.");
  console.log(`Output directory: ${outputDir}`);
  try {
    console.log("[1/5] Loading fixed, dynamic, and Bridge result artifacts...");
    const loaded = {};
    for (const [key, sourcePath] of Object.entries(inputPaths)) {
      loaded[key] = typedRows(parseCsv(await fs.readFile(sourcePath, "utf8")));
    }
    const bridgeEventHeader = loaded.bridgePerEvent[0];
    const eventColumn = bridgeEventHeader.indexOf("event_id");
    if (eventColumn < 0) throw new Error("Bridge per-event input has no event_id column");
    loaded.bridgePerEvent = [
      bridgeEventHeader,
      ...loaded.bridgePerEvent.slice(1).filter((row) => primaryEvents.has(String(row[eventColumn]))),
    ];

    console.log("[2/5] Creating summary and result sheets...");
    const workbook = Workbook.create();
    const guide = workbook.worksheets.add("Guide");
    guide.showGridLines = false;
    guide.getRange("A2:H2").merge();
    guide.getRange("A2").values = [["PHEME three-stage intervention evaluation"]];
    guide.getRange("A2:H2").format.font = { name: "Arial", size: 16, bold: true, color: "#1F2937" };
    guide.getRange("A4:D8").values = [
      ["Stage", "Purpose", "Numerator", "Denominator"],
      ["Fixed 30 min", "Compare fixed-30 ranking models", "Selected PI(30)", "Candidate PI(30)"],
      ["Dynamic", "Compare sequential intervention policies", "Selected PI(T_i)", "Candidate PI(10), counted once"],
      ["Bridge", "Compare waiting until 30 min against dynamic intervention", "Fixed PI(30) or dynamic PI(T_i)", "Common candidate PI(10)"],
      ["Aggregation", "Primary: pooled; secondary: macro", "Sum blocked PI", "Matching opportunity PI"],
    ];
    guide.getRange("A4:D4").format = { fill: "#1F4E78", font: { name: "Arial", bold: true, color: "#FFFFFF" } };
    guide.getRange("A4:D8").format.font = { name: "Arial", size: 10 };
    guide.getRange("A4:D8").format.wrapText = true;
    guide.getRange("A4:D8").format.verticalAlignment = "center";
    guide.getRange("A4:D8").format.autofitRows();
    guide.getRange("A:D").format.columnWidth = 30;
    guide.getRange("A10:H13").values = [
      ["Interpretation notes", "", "", "", "", "", "", ""],
      ["Pooled is the primary metric because the research objective is to block the largest total number of future nodes.", "", "", "", "", "", "", ""],
      ["Macro gives every event equal weight and is retained to assess cross-event stability.", "", "", "", "", "", "", ""],
      ["Only the Bridge sheet is a direct fixed-versus-dynamic comparison; native Fixed and Dynamic scores answer different questions.", "", "", "", "", "", "", ""],
    ];
    guide.getRange("A10:H10").merge();
    guide.getRange("A11:H11").merge(); guide.getRange("A12:H12").merge(); guide.getRange("A13:H13").merge();
    guide.getRange("A10:H10").format = { fill: "#D9EAF7", font: { name: "Arial", bold: true, color: "#1F2937" } };
    guide.getRange("A11:H13").format.font = { name: "Arial", size: 10, italic: true, color: "#4B5563" };

    const fixed = workbook.worksheets.add("Fixed results");
    const dynamic = workbook.worksheets.add("Dynamic results");
    const checkpoints = workbook.worksheets.add("Dynamic checkpoints");
    const bridge = workbook.worksheets.add("Bridge results");
    const bridgeEvents = workbook.worksheets.add("Bridge per event");
    styleDataSheet(fixed, loaded.fixed, "FixedResultsTable");
    styleDataSheet(dynamic, loaded.dynamic, "DynamicResultsTable");
    styleDataSheet(checkpoints, loaded.dynamicCheckpoint, "DynamicCheckpointTable");
    styleDataSheet(bridge, loaded.bridge, "BridgeResultsTable");
    styleDataSheet(bridgeEvents, loaded.bridgePerEvent, "BridgePerEventTable");

    console.log("[3/5] Applying research formats and recalculating...");
    workbook.recalculate();

    console.log("[4/5] Inspecting and rendering every worksheet...");
    const checks = {};
    for (const name of ["Guide", "Fixed results", "Dynamic results", "Dynamic checkpoints", "Bridge results", "Bridge per event"]) {
      const sheet = workbook.worksheets.getItem(name);
      const inspection = await workbook.inspect({ kind: "table", sheetId: sheet.sheetId, maxChars: 2000, tableMaxRows: 5, tableMaxCols: 12 });
      checks[name] = inspection.ndjson;
      const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
      await fs.writeFile(path.join(outputDir, `preview_${name.replaceAll(" ", "_").toLowerCase()}.png`), new Uint8Array(await preview.arrayBuffer()));
    }
    const errors = await workbook.inspect({
      kind: "match",
      searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
      options: { useRegex: true, maxResults: 300 },
      summary: "final formula error scan",
    });

    console.log("[5/5] Exporting workbook and completing run record...");
    const workbookPath = path.join(outputDir, "three_stage_fixed_dynamic_bridge_results.xlsx");
    const output = await SpreadsheetFile.exportXlsx(workbook);
    await output.save(workbookPath);
    runRecord.status = "complete";
    runRecord.completed_at = isoNow();
    runRecord.metrics_results = {
      fixed_rows: loaded.fixed.length - 1,
      dynamic_rows: loaded.dynamic.length - 1,
      dynamic_checkpoint_rows: loaded.dynamicCheckpoint.length - 1,
      bridge_rows: loaded.bridge.length - 1,
      bridge_per_event_rows: loaded.bridgePerEvent.length - 1,
      formula_error_scan: errors.ndjson,
      inspection_completed_for_all_sheets: true,
    };
    runRecord.output_file_inventory = (await fs.readdir(outputDir)).sort();
    await fs.writeFile(runRecordPath, JSON.stringify(runRecord, null, 2));
    console.log(`SUCCESS: three-stage result workbook saved to: ${outputDir}`);
  } catch (error) {
    runRecord.status = "failed";
    runRecord.completed_at = isoNow();
    runRecord.failure_details = { type: error?.name ?? "Error", message: String(error?.message ?? error), stack: error?.stack ?? null };
    runRecord.output_file_inventory = (await fs.readdir(outputDir)).sort();
    await fs.writeFile(runRecordPath, JSON.stringify(runRecord, null, 2));
    console.error(`FAILURE: three-stage result workbook preserved at: ${outputDir}`);
    throw error;
  }
}

await main();
