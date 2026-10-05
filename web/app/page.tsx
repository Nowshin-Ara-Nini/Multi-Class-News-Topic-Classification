"use client";

import { useEffect, useState } from "react";

type Prediction = { text: string; predicted_class: string; confidence: number; probabilities: Record<string, number>; inference_time_ms: number };
type Card = { model_name: string; architecture: string; preprocessing_mode: string; artifact_sha256: string; quantized: boolean;
  training_summary: { best_val_f1?: number; best_val_acc?: number };
  test_metrics: { f1_macro?: number; accuracy?: number; test_rows?: number; macro_f1_ci_low?: number; macro_f1_ci_high?: number } | null };
type Comparison = { family: string; baseline_val_f1: number | null; val_f1: number; test_f1: number | null; selected: boolean };
const examples = [
  ["Business", "Apple reports record quarterly revenue as iPhone sales rise"],
  ["Science & technology", "Scientists discover a new planet orbiting a nearby star"],
  ["Sports", "Argentina wins the final after a dramatic penalty shootout"],
  ["World news", "World leaders meet to discuss a new international peace agreement"],
];
const percent = (n: number | null | undefined) => n == null ? "Not evaluated" : `${(n * 100).toFixed(2)}%`;
async function request(path: string, data?: unknown) {
  const response = await fetch(`/api/${path}`, data === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const json = await response.json();
  if (!response.ok) throw new Error(json.error || "Request failed. Please retry.");
  return json;
}

export default function Home() {
  const [tab, setTab] = useState("classify");
  const [text, setText] = useState("");
  const [batch, setBatch] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [rows, setRows] = useState<Prediction[]>([]);
  const [card, setCard] = useState<Card | null>(null);
  const [models, setModels] = useState<Comparison[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connection, setConnection] = useState("Connecting to the model…");
  const [progress, setProgress] = useState("");

  async function connect() {
    setConnection("Connecting to the model…");
    try {
      setCard(await request("model/info"));
      setConnection("Model ready");
      const comparison = await request("models");
      setModels(comparison.models);
    } catch (e) { setCard(null); setConnection((e as Error).message); }
  }
  useEffect(() => { void connect(); }, []);

  async function classify() {
    setBusy(true); setError(""); setResult(null);
    try { setResult((await request("predict", { text })).result); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function classifyBatch() {
    const texts = batch.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!texts.length || texts.length > 100 || texts.some((line) => line.length > 5000)) {
      setError("Enter 1–100 headlines, each at most 5,000 characters."); return;
    }
    setBusy(true); setError(""); setRows([]);
    const collected: Prediction[] = [];
    try {
      for (let start = 0; start < texts.length; start += 10) {
        setProgress(`Classifying ${start + 1}–${Math.min(start + 10, texts.length)} of ${texts.length}`);
        collected.push(...(await request("batch_predict", { texts: texts.slice(start, start + 10) })).results);
        setRows([...collected]);
      }
    } catch (e) { setError(`${(e as Error).message} ${collected.length} results completed.`); }
    finally { setBusy(false); setProgress(""); }
  }

  return <main>
    <header><a className="brand" href="/">THE TOPIC DESK<span>NEWS, IN CONTEXT.</span></a><span className="edition">A MACHINE LEARNING PROJECT <i>●</i> FOUR TOPICS</span></header>
    <section className="intro"><p className="eyebrow">FROM HEADLINE TO UNDERSTANDING</p><h1>Every story has<br />a <em>point of view.</em></h1>
      <p className="lede">Find where a headline belongs. Explore predictions from a trained news classifier, with the numbers behind the model.</p>
      <div className="topics">{examples.map(([label], i) => <span key={label}><b>0{i + 1}</b>{label}</span>)}</div>
    </section>
    <div className="workspace">
      <section className="desk">
        <nav aria-label="Workspace">{[["classify", "Classify a story"], ["batch", "Batch desk"], ["models", "Model results"]].map(([value, label]) =>
          <button key={value} className={tab === value ? "active" : ""} onClick={() => { setTab(value); setError(""); }}>{label}</button>)}</nav>
        {tab === "classify" && <div className="panel">
          <div className="section-heading"><span className="eyebrow">01 / CLASSIFICATION</span><span>ONE HEADLINE, FOUR POSSIBILITIES</span></div>
          <label htmlFor="headline"><h2>What’s the story?</h2></label><p className="muted">Paste a headline or a short news excerpt.</p>
          <textarea id="headline" value={text} maxLength={5000} onChange={(e) => setText(e.target.value)} placeholder="A new discovery changes how scientists understand the universe…" />
          <div className="input-footer"><span>{text.length.toLocaleString()} / 5,000 characters</span><button className="primary" disabled={busy || !text.trim()} onClick={classify}>{busy ? "Reading the story…" : "Classify headline ↗"}</button></div>
          <div className="examples"><span>TRY A STORY</span>{examples.map(([label, headline]) => <button key={label} onClick={() => { setText(headline); setResult(null); }}>{label} ↗</button>)}</div>
          {result && <section className="prediction" aria-live="polite"><p className="eyebrow">THE MODEL’S PREDICTION</p><h2>{result.predicted_class}</h2><p>{percent(result.confidence)} model confidence</p>
            <div className="bars">{Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).map(([label, value]) => <div key={label}><div className="bar-label"><span>{label}</span><b>{percent(value)}</b></div><div className="track"><div style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div></div>)}</div>
            <small>Confidence is the model’s score for this prediction, not its measured test accuracy.</small></section>}
        </div>}
        {tab === "batch" && <div className="panel"><p className="eyebrow">02 / BATCH CLASSIFICATION</p><h2>A desk full of stories.</h2><p className="muted">One headline per line. Up to 100 stories, processed ten at a time.</p>
          <label className="sr-only" htmlFor="batch">Headlines, one per line</label><textarea id="batch" value={batch} onChange={(e) => setBatch(e.target.value)} placeholder="Paste your headlines here, one per line…" />
          <button className="primary" onClick={classifyBatch} disabled={busy || !batch.trim()}>{busy ? progress : "Classify batch ↗"}</button>
          {!!rows.length && <div className="table-wrap"><table><thead><tr><th>Headline</th><th>Topic</th><th>Confidence</th></tr></thead><tbody>{rows.map((r, i) => <tr key={i}><td>{r.text}</td><td>{r.predicted_class}</td><td>{percent(r.confidence)}</td></tr>)}</tbody></table></div>}
        </div>}
        {tab === "models" && <div className="panel"><p className="eyebrow">03 / MEASURED PERFORMANCE</p><h2>Show the evidence.</h2><p className="muted">Models are selected by validation macro F1. The supplied test CSV is used only for explicit final evaluation.</p>
          {models.length ? <div className="table-wrap"><table><thead><tr><th>Model family</th><th>Baseline val F1</th><th>Selected val F1</th><th>Test F1</th></tr></thead><tbody>{models.map((m) => <tr key={m.family}><td>{m.family}{m.selected && <span className="badge">SELECTED</span>}</td><td>{percent(m.baseline_val_f1)}</td><td>{percent(m.val_f1)}</td><td>{percent(m.test_f1)}</td></tr>)}</tbody></table></div> : <p className="empty">Comparison results will appear when an evaluated model is deployed.</p>}
          <p className="footnote">The table reports original family checkpoints. The sidebar reports the exact deployed artifact, which may be quantized.</p>
        </div>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>
      <aside><p className="eyebrow">AT THE EDITOR’S DESK</p><h2>The model<br />behind the story.</h2><div className="status"><span className={card ? "dot" : "dot waiting"} />{connection}</div><button className="text-button" onClick={connect}>Refresh connection ↻</button>
        <dl><dt>DEPLOYED MODEL</dt><dd>{card?.architecture || "Awaiting deployment"}</dd><dt>PREPROCESSING</dt><dd>{card?.preprocessing_mode || "—"}</dd><dt>VALIDATION MACRO F1</dt><dd className="score">{percent(card?.training_summary.best_val_f1)}</dd><dt>TEST MACRO F1</dt><dd className="score accent">{percent(card?.test_metrics?.f1_macro)}</dd><dt>TEST ACCURACY</dt><dd>{percent(card?.test_metrics?.accuracy)}</dd><dt>TEST EXAMPLES</dt><dd>{card?.test_metrics?.test_rows?.toLocaleString() || "Not evaluated"}</dd></dl>
        {card?.artifact_sha256 && <p className="version">ARTIFACT {card.artifact_sha256.slice(0, 12)}{card.quantized ? " · INT8" : ""}</p>}
        <p className="aside-note">One selected model serves predictions. No training or model selection happens on this website.</p>
      </aside>
    </div>
    <footer><div className="footer-credit"><p>© 2026 NOWSHIN ARA NINI. All rights reserved.</p><p>Developed by <a href="https://github.com/Nowshin-Ara-Nini" target="_blank" rel="noreferrer">Nowshin Ara Nini</a></p></div></footer>
  </main>;
}
