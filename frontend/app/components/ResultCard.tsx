"use client";

import { AlertTriangle, CheckCircle2, Info } from "lucide-react";
import type { PredictionResult } from "../page";
import { WordHighlight } from "./WordHighlight";
import { ConfidenceGauge } from "./ConfidenceGauge";

type Props = {
  result: PredictionResult | null;
  inputText: string;
  isLoading: boolean;
};

export function ResultCard({ result, inputText, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="glass-card p-5 min-h-[340px] flex flex-col items-center justify-center gap-3">
        <div className="w-10 h-10 border-2 border-border border-t-primary rounded-full animate-spin" />
        <p className="text-sm text-muted">Analyzing article…</p>
        <div className="flex gap-1 mt-1">
          {["Tokenizing", "Embedding", "Classifying"].map((step, i) => (
            <span key={step} className="text-xs text-muted/60 font-mono" style={{ animationDelay: `${i * 0.2}s` }}>
              {step}
            </span>
          ))}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="glass-card p-5 min-h-[340px] flex flex-col items-center justify-center gap-4 text-center">
        <div className="w-14 h-14 rounded-full border-2 border-dashed border-border flex items-center justify-center">
          <Info className="w-6 h-6 text-muted" />
        </div>
        <div>
          <p className="text-sm font-medium text-foreground mb-1">No analysis yet</p>
          <p className="text-xs text-muted leading-relaxed">
            Paste a news article and click &ldquo;Analyze Article&rdquo; to get a FAKE or REAL verdict
            with word-level explanations.
          </p>
        </div>
      </div>
    );
  }

  const isFake = result.label === "FAKE";
  const pct = Math.round(result.confidence * 100);

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Verdict banner */}
      <div
        className={`glass-card p-5 border ${
          isFake ? "border-danger/40 bg-danger/5" : "border-success/40 bg-success/5"
        }`}
      >
        <div className="flex items-start gap-3 mb-4">
          <div
            className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center ${
              isFake ? "bg-danger/20 text-danger" : "bg-success/20 text-success"
            }`}
          >
            {isFake ? (
              <AlertTriangle className="w-6 h-6" />
            ) : (
              <CheckCircle2 className="w-6 h-6" />
            )}
          </div>
          <div className="flex-1">
            <p className="text-xs text-muted mb-0.5">AI Verdict</p>
            <h3
              className={`text-2xl font-bold ${isFake ? "text-danger" : "text-success"}`}
            >
              {isFake ? "LIKELY FAKE" : "LIKELY REAL"}
            </h3>
            <p className="text-xs text-muted mt-1 leading-relaxed">
              {isFake 
                ? "This article contains patterns commonly associated with misinformation, including sensational language, unverified claims, or conspiracy-style rhetoric."
                : "This article exhibits characteristics of credible journalism, including factual reporting, cited sources, and measured language."
              }
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-muted mb-0.5">Confidence</p>
            <p className={`text-3xl font-bold tabular-nums ${isFake ? "text-danger" : "text-success"}`}>
              {pct}%
            </p>
          </div>
        </div>

        <ConfidenceGauge fake={result.probabilities.FAKE} real={result.probabilities.REAL} />

        {/* Meta */}
        <div className="mt-4 pt-3 border-t border-border/50 flex flex-wrap gap-4 text-xs text-muted">
          <span>
            Model:{" "}
            <span className="text-foreground font-medium">{result.model_used}</span>
          </span>
          <span>
            Analyzed:{" "}
            <span className="text-foreground font-medium">{result.word_count} words</span>
          </span>
          <span>
            Length:{" "}
            <span className="text-foreground font-medium">{result.char_count} chars</span>
          </span>
        </div>
      </div>

      {/* Word importance */}
      {result.word_importance.length > 0 && (
        <WordHighlight
          wordImportance={result.word_importance}
          inputText={inputText}
        />
      )}
    </div>
  );
}
