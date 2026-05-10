"use client";

import { useState } from "react";
import { Send, Trash2, ChevronDown } from "lucide-react";
import type { PredictionResult } from "../page";

type Props = {
  onResult: (result: PredictionResult) => void;
  onTextChange: (text: string) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
};

const MODELS = [
  { value: "ensemble", label: "Ensemble (Recommended)", badge: "Best" },
  { value: "tfidf_lr", label: "TF-IDF + Logistic Regression", badge: "Fast" },
  { value: "lstm", label: "LSTM Neural Network", badge: null },
  { value: "bilstm", label: "Bidirectional LSTM", badge: null },
  { value: "bert", label: "BERT Transformer", badge: "SOTA" },
];

export function PredictionForm({ onResult, onTextChange, isLoading, setIsLoading }: Props) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("ensemble");
  const [error, setError] = useState<string | null>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const charCount = text.length;

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    onTextChange(e.target.value);
  };

  const handleClear = () => {
    setText("");
    onTextChange("");
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || text.length < 10) {
      setError("Please enter at least 10 characters of news text.");
      return;
    }
    setError(null);
    setIsLoading(true);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), model }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const data: PredictionResult = await res.json();
      onResult(data);
    } catch (err: any) {
      setError(err.message || "Prediction failed. Make sure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="glass-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-foreground">News Article</h2>
        <span className="text-xs text-muted font-mono">
          {wordCount} words &bull; {charCount} chars
        </span>
      </div>

      {/* Textarea */}
      <div className="relative">
        <textarea
          value={text}
          onChange={handleChange}
          placeholder="Paste or type a news article here…

Example: Scientists at MIT have developed a revolutionary solar panel that achieves record-breaking efficiency of 47%, potentially transforming renewable energy markets worldwide..."
          rows={9}
          className="w-full bg-surface-2 border border-border rounded-lg px-4 py-3 text-sm text-foreground placeholder-muted/60 resize-none outline-none focus:border-primary/60 transition-colors leading-relaxed font-sans"
          disabled={isLoading}
        />
        {text.length > 0 && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute top-3 right-3 text-muted hover:text-foreground transition-colors"
            title="Clear text"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Model selector */}
      <div className="space-y-1.5">
        <label className="text-xs text-muted font-medium">Model</label>
        <div className="relative">
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full appearance-none bg-surface-2 border border-border rounded-lg px-4 py-2.5 text-sm text-foreground outline-none focus:border-primary/60 transition-colors cursor-pointer"
            disabled={isLoading}
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}{m.badge ? ` — ${m.badge}` : ""}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none" />
        </div>
      </div>

      {/* Error */}
      {error && (
        <p className="text-xs text-danger bg-danger/10 border border-danger/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={isLoading || !text.trim()}
        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground font-medium text-sm transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <>
            <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Send className="w-4 h-4" />
            Analyze Article
          </>
        )}
      </button>
    </form>
  );
}
