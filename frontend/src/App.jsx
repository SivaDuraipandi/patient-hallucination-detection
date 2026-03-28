import { useMemo, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const sampleCase = {
  patient_context:
    "54-year-old patient with persistent fever, productive cough, pleuritic chest pain, and elevated inflammatory markers after three days of worsening symptoms. Oxygen saturation is mildly reduced, and chest examination suggests lower respiratory tract involvement.",
  question:
    "Is this presentation likely bacterial pneumonia requiring antibiotic treatment?",
  answer:
    "This is almost certainly viral and does not require antibiotics because chest pain does not happen with bacterial pneumonia.",
};

const initialForm = {
  patient_context: "",
  question: "",
  answer: "",
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function App() {
  const [theme, setTheme] = useState("light");
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({
    tone: "neutral",
    message: "Enter a case and run the analysis.",
  });
  const [result, setResult] = useState(null);

  const resultTone = useMemo(() => {
    if (!result) {
      return "neutral";
    }
    return result.is_hallucinated ? "danger" : "safe";
  }, [result]);

  function updateField(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  function loadSample() {
    setForm(sampleCase);
    setStatus({
      tone: "neutral",
      message: "Sample loaded. You can edit it before analysis.",
    });
  }

  function clearAll() {
    setForm(initialForm);
    setResult(null);
    setStatus({
      tone: "neutral",
      message: "Enter a case and run the analysis.",
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const payload = {
      patient_context: form.patient_context.trim(),
      question: form.question.trim(),
      answer: form.answer.trim(),
    };

    if (!payload.patient_context || !payload.question || !payload.answer) {
      setStatus({
        tone: "danger",
        message: "Please complete all fields before analysis.",
      });
      return;
    }

    setLoading(true);
    setStatus({
      tone: "loading",
      message: "Analyzing the response with the MiniLM model...",
    });

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Backend request failed");
      }

      const data = await response.json();
      setResult(data);
      setStatus({
        tone: data.is_hallucinated ? "danger" : "safe",
        message: data.is_hallucinated
          ? "Potential hallucination detected. Review this answer carefully."
          : "The answer looks acceptable according to the current model.",
      });
    } catch (error) {
      setResult(null);
      setStatus({
        tone: "danger",
        message:
          `Unable to reach the backend at ${API_BASE_URL}. ` +
          `Make sure FastAPI is running. Error: ${error.message}`,
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`app-shell theme-${theme}`}>
      <div className="page">
        <header className="topbar">
          <div>
            <p className="kicker">Clinical AI Review</p>
            <h1>Patient Hallucination Detection</h1>
            <p className="subtitle">
              A simple review workspace for checking whether a medical answer is likely hallucinated.
            </p>
          </div>

          <div className="topbar-actions">
            <div className="badge">
              <span>Model</span>
              <strong>MiniLM-L12-H384</strong>
            </div>
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
            >
              {theme === "light" ? "Dark Mode" : "Light Mode"}
            </button>
          </div>
        </header>

        <main className="layout">
          <section className="card input-card">
            <div className="card-header">
              <div>
                <p className="section-label">Input</p>
                <h2>Case Details</h2>
              </div>
              <div className="header-buttons">
                <button type="button" className="ghost-button" onClick={loadSample}>
                  Load Sample
                </button>
                <button type="button" className="ghost-button" onClick={clearAll}>
                  Clear
                </button>
              </div>
            </div>

            <form className="form-grid" onSubmit={handleSubmit}>
              <label className="field">
                <span>Patient Context</span>
                <textarea
                  name="patient_context"
                  rows="7"
                  value={form.patient_context}
                  onChange={updateField}
                  placeholder="Paste the patient history or clinical summary here..."
                  required
                />
              </label>

              <label className="field">
                <span>Question</span>
                <textarea
                  name="question"
                  rows="3"
                  value={form.question}
                  onChange={updateField}
                  placeholder="Enter the medical question..."
                  required
                />
              </label>

              <label className="field">
                <span>Candidate Answer</span>
                <textarea
                  name="answer"
                  rows="4"
                  value={form.answer}
                  onChange={updateField}
                  placeholder="Paste the answer you want the model to review..."
                  required
                />
              </label>

              <div className="submit-row">
                <button type="submit" className="primary-button" disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze Answer"}
                </button>
              </div>
            </form>
          </section>

          <section className="card result-card">
            <div className="card-header">
              <div>
                <p className="section-label">Output</p>
                <h2>Model Result</h2>
              </div>
              <div className={`status-pill ${status.tone}`}>{status.tone}</div>
            </div>

            <div className={`status-panel ${status.tone}`}>
              {status.message}
            </div>

            <div className={`result-hero ${resultTone}`}>
              <span className="result-overline">Prediction</span>
              <h3>
                {result
                  ? result.is_hallucinated
                    ? "Hallucinated"
                    : "Not Hallucinated"
                  : "Awaiting Analysis"}
              </h3>
              <p>
                {result
                  ? result.is_hallucinated
                    ? "The response appears risky and should be reviewed."
                    : "The response appears acceptable under the current model."
                  : "Submit a case to generate a prediction and trust score."}
              </p>
            </div>

            <div className="metric-grid">
              <article className="metric-card">
                <span>Hallucination Probability</span>
                <strong>
                  {result ? formatPercent(result.hallucination_probability) : "--"}
                </strong>
              </article>
              <article className="metric-card">
                <span>Confidence</span>
                <strong>{result ? formatPercent(result.confidence) : "--"}</strong>
              </article>
              <article className="metric-card">
                <span>Trust Score</span>
                <strong>{result ? result.trust_score.toFixed(4) : "--"}</strong>
              </article>
              <article className="metric-card">
                <span>Uncertainty</span>
                <strong>{result ? formatPercent(result.uncertainty) : "--"}</strong>
              </article>
              <article className="metric-card">
                <span>Neighbor Trust</span>
                <strong>{result ? result.neighbor_trust.toFixed(4) : "--"}</strong>
              </article>
              <article className="metric-card">
                <span>Review Flag</span>
                <strong>{result ? (result.abstain_for_review ? "Review" : "Clear") : "--"}</strong>
              </article>
              <article className="metric-card">
                <span>Runtime Device</span>
                <strong>{result ? result.device : "--"}</strong>
              </article>
            </div>

            <div className="assistant-note">
              <h4>Clinical Note</h4>
              <p>
                {result
                  ? result.is_hallucinated
                    ? `This answer was flagged with ${formatPercent(result.hallucination_probability)} hallucination probability, uncertainty ${formatPercent(result.uncertainty)}, and trust score ${result.trust_score.toFixed(4)}.`
                    : `This answer was accepted with calibrated confidence ${formatPercent(result.calibrated_probability)} and trust score ${result.trust_score.toFixed(4)}.`
                  : "The backend will return a calibrated prediction, uncertainty, neighbor trust, and final trust score after analysis."}
              </p>
              {result ? (
                <p>
                  Signals: {result.explanation_tags.join(", ")}
                </p>
              ) : null}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
