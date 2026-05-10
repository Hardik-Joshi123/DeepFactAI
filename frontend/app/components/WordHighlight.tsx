"use client";

import { useState } from "react";
import { Eye, BarChart2 } from "lucide-react";

type WordImp = {
  word: string;
  score: number;
  fake_contribution: boolean;
};

type Props = {
  wordImportance: WordImp[];
  inputText: string;
};

function highlightText(text: string, wordImportance: WordImp[]): React.ReactNode[] {
  if (!wordImportance.length) return [text];

  const scoreMap = new Map<string, WordImp>(
    wordImportance.map((w) => [w.word.toLowerCase(), w])
  );

  const words = text.split(/(\s+)/);
  return words.map((token, i) => {
    const clean = token.toLowerCase().replace(/[^a-z0-9]/g, "");
    const imp = scoreMap.get(clean);
    if (imp && Math.abs(imp.score) > 0.001) {
      const intensity = Math.min(Math.abs(imp.score) * 8, 0.85);
      const bg = imp.fake_contribution
        ? `rgba(239,68,68,${intensity.toFixed(2)})`
        : `rgba(34,197,94,${intensity.toFixed(2)})`;
      return (
        <mark
          key={i}
          title={`Score: ${imp.score.toFixed(3)} — ${imp.fake_contribution ? "FAKE" : "REAL"} indicator`}
          style={{
            backgroundColor: bg,
            borderRadius: "2px",
            padding: "1px 2px",
            cursor: "default",
          }}
          className="text-foreground font-medium"
        >
          {token}
        </mark>
      );
    }
    return token;
  });
}

export function WordHighlight({ wordImportance, inputText }: Props) {
  const [view, setView] = useState<"text" | "chart">("chart");

  const sorted = [...wordImportance].sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const maxAbs = Math.max(...sorted.map((w) => Math.abs(w.score)), 0.001);

  return (
    <div className="glass-card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">Word Importance</h3>
        <div className="flex rounded-md border border-border overflow-hidden text-xs">
          <button
            onClick={() => setView("chart")}
            className={`flex items-center gap-1 px-2.5 py-1.5 transition-colors ${
              view === "chart"
                ? "bg-primary/20 text-primary"
                : "text-muted hover:text-foreground"
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Chart
          </button>
          <button
            onClick={() => setView("text")}
            className={`flex items-center gap-1 px-2.5 py-1.5 transition-colors ${
              view === "text"
                ? "bg-primary/20 text-primary"
                : "text-muted hover:text-foreground"
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            Highlight
          </button>
        </div>
      </div>

      {/* Legend and explanation */}
      <div className="bg-surface-2/50 rounded-lg p-3 space-y-2">
        <div className="flex gap-4 text-xs text-muted">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-danger/60 inline-block" />
            Fake News Indicator
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-success/60 inline-block" />
            Credible News Indicator
          </div>
        </div>
        <p className="text-xs text-muted/80 leading-relaxed">
          Words highlighted in <span className="text-danger">red</span> are commonly found in misinformation (sensationalism, conspiracy language). 
          Words in <span className="text-success">green</span> indicate credible journalism (citations, statistics, official sources).
        </p>
      </div>

      {/* Chart view */}
      {view === "chart" && (
        <div className="space-y-2">
          {sorted.slice(0, 10).map((item) => {
            const barPct = (Math.abs(item.score) / maxAbs) * 100;
            return (
              <div key={item.word} className="flex items-center gap-2">
                <span className="w-24 text-xs text-muted text-right truncate font-mono">
                  {item.word}
                </span>
                <div className="flex-1 h-5 bg-surface-2 rounded-sm overflow-hidden">
                  <div
                    className={`h-full rounded-sm transition-all duration-500 ${
                      item.fake_contribution ? "bg-danger/70" : "bg-success/70"
                    }`}
                    style={{ width: `${barPct}%` }}
                  />
                </div>
                <span className="w-14 text-xs text-muted text-right font-mono">
                  {item.score.toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Highlight view */}
      {view === "text" && (
        <div className="text-sm leading-relaxed text-foreground bg-surface-2 rounded-lg p-4 max-h-52 overflow-y-auto">
          {inputText
            ? highlightText(inputText, wordImportance)
            : <span className="text-muted">No article text available.</span>
          }
        </div>
      )}
    </div>
  );
}
