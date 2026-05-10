"use client";

const STEPS = [
  {
    number: "01",
    title: "Text Preprocessing",
    description:
      "The input article is cleaned: HTML tags, URLs and special characters are removed. NLTK tokenises the text and WordNet lemmatiser reduces words to their base form.",
    tech: ["NLTK", "Lemmatization", "Tokenization"],
  },
  {
    number: "02",
    title: "Feature Extraction",
    description:
      "For deep learning models, a vocabulary is built and each word is mapped to a dense 300-dimensional GloVe vector. For TF-IDF, word and character n-grams are extracted with sublinear TF scaling.",
    tech: ["GloVe-300d", "TF-IDF", "n-grams"],
  },
  {
    number: "03",
    title: "Deep Learning Inference",
    description:
      "Stacked LSTM / BiLSTM layers with an attention mechanism capture sequential patterns. Attention weights highlight the most influential words. BERT processes the full sequence with bidirectional context.",
    tech: ["LSTM", "BiLSTM", "BERT", "Attention"],
  },
  {
    number: "04",
    title: "Ensemble & Explainability",
    description:
      "Soft-voting combines predictions from all models. Word importance is derived from LR coefficients, attention weights, and LIME perturbations — mapping each word's contribution to the FAKE/REAL decision.",
    tech: ["Ensemble", "LIME", "Attention Weights"],
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-foreground">How It Works</h2>
        <p className="text-sm text-muted mt-0.5">
          End-to-end deep learning pipeline from raw text to verdict
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {STEPS.map((step, i) => (
          <div key={step.number} className="glass-card p-5 space-y-3 relative overflow-hidden">
            {/* Connector line */}
            {i < STEPS.length - 1 && (
              <div className="hidden lg:block absolute top-8 right-0 translate-x-1/2 w-4 border-t border-dashed border-border z-10" />
            )}

            <div className="flex items-center gap-2">
              <span className="text-3xl font-bold font-mono text-primary/20">{step.number}</span>
            </div>
            <h3 className="font-medium text-foreground text-sm">{step.title}</h3>
            <p className="text-xs text-muted leading-relaxed">{step.description}</p>
            <div className="flex flex-wrap gap-1.5">
              {step.tech.map((t) => (
                <span
                  key={t}
                  className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/15 font-mono"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
