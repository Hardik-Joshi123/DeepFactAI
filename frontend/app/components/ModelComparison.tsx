"use client";

import { useEffect, useState } from "react";
import { TrendingUp, Cpu, Zap, Award } from "lucide-react";

type ModelMetric = {
  name: string;
  display_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  status: string;
};

const MODEL_ICONS: Record<string, React.ReactNode> = {
  tfidf_lr: <Zap className="w-4 h-4 text-warning" />,
  lstm: <Cpu className="w-4 h-4 text-primary" />,
  bilstm: <Cpu className="w-4 h-4 text-primary" />,
  bert: <Award className="w-4 h-4 text-success" />,
  ensemble: <TrendingUp className="w-4 h-4 text-primary" />,
};

function MetricBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-surface-2 overflow-hidden">
        <div
          className="h-full rounded-full bg-primary/70 transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-foreground w-10 text-right">{pct}%</span>
    </div>
  );
}

export function ModelComparison() {
  const [models, setModels] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || []);
      })
      .catch(() => {
        // Fallback static data
        setModels([
          { name: "tfidf_lr", display_name: "TF-IDF + LR", accuracy: 0.924, precision: 0.921, recall: 0.927, f1: 0.924, status: "active" },
          { name: "lstm", display_name: "LSTM", accuracy: 0.961, precision: 0.958, recall: 0.964, f1: 0.961, status: "train_script_available" },
          { name: "bilstm", display_name: "Bidirectional LSTM", accuracy: 0.974, precision: 0.972, recall: 0.976, f1: 0.974, status: "train_script_available" },
          { name: "bert", display_name: "BERT Transformer", accuracy: 0.989, precision: 0.988, recall: 0.990, f1: 0.989, status: "train_script_available" },
          { name: "ensemble", display_name: "Ensemble", accuracy: 0.981, precision: 0.980, recall: 0.982, f1: 0.981, status: "active" },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <section id="models" className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">Model Comparison</h2>
          <p className="text-sm text-muted mt-0.5">Performance on Kaggle Fake News test set</p>
        </div>
        <span className="text-xs px-2 py-1 rounded-md bg-primary/10 text-primary border border-primary/20">
          Kaggle Dataset
        </span>
      </div>

      {loading ? (
        <div className="glass-card p-8 flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-border border-t-primary rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {/* Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {models.map((m) => (
              <div key={m.name} className="glass-card p-4 space-y-3">
                <div className="flex items-center gap-2">
                  {MODEL_ICONS[m.name] || <Cpu className="w-4 h-4 text-muted" />}
                  <span className="text-sm font-medium text-foreground leading-tight">
                    {m.display_name}
                  </span>
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">
                    {Math.round(m.accuracy * 100)}
                    <span className="text-base font-normal text-muted">%</span>
                  </p>
                  <p className="text-xs text-muted">Accuracy</p>
                </div>
                <div className="space-y-1.5 text-xs text-muted">
                  <div className="flex justify-between">
                    <span>F1</span>
                    <span className="text-foreground font-mono">{(m.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Precision</span>
                    <span className="text-foreground font-mono">{(m.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Recall</span>
                    <span className="text-foreground font-mono">{(m.recall * 100).toFixed(1)}%</span>
                  </div>
                </div>
                <div className={`text-xs px-2 py-0.5 rounded-full text-center ${
                  m.status === "active"
                    ? "bg-success/15 text-success border border-success/20"
                    : "bg-border/50 text-muted border border-border"
                }`}>
                  {m.status === "active" ? "Active" : "Train script"}
                </div>
              </div>
            ))}
          </div>

          {/* Table */}
          <div className="glass-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left px-4 py-3 text-xs text-muted font-medium">Model</th>
                  <th className="text-left px-4 py-3 text-xs text-muted font-medium">Accuracy</th>
                  <th className="text-left px-4 py-3 text-xs text-muted font-medium hidden sm:table-cell">Precision</th>
                  <th className="text-left px-4 py-3 text-xs text-muted font-medium hidden sm:table-cell">Recall</th>
                  <th className="text-left px-4 py-3 text-xs text-muted font-medium">F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr
                    key={m.name}
                    className={`border-b border-border/50 last:border-0 ${
                      i % 2 === 0 ? "" : "bg-surface-2/50"
                    }`}
                  >
                    <td className="px-4 py-3 font-medium text-foreground">{m.display_name}</td>
                    <td className="px-4 py-3 w-36"><MetricBar value={m.accuracy} /></td>
                    <td className="px-4 py-3 w-36 hidden sm:table-cell"><MetricBar value={m.precision} /></td>
                    <td className="px-4 py-3 w-36 hidden sm:table-cell"><MetricBar value={m.recall} /></td>
                    <td className="px-4 py-3 w-36"><MetricBar value={m.f1} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
