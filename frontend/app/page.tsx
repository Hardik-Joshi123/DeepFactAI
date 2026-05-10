"use client";

import { useState } from "react";
import { Header } from "./components/Header";
import { PredictionForm } from "./components/PredictionForm";
import { ResultCard } from "./components/ResultCard";
import { ModelComparison } from "./components/ModelComparison";
import { HowItWorks } from "./components/HowItWorks";
import { SampleNews } from "./components/SampleNews";

export type PredictionResult = {
  label: "FAKE" | "REAL";
  confidence: number;
  probabilities: { FAKE: number; REAL: number };
  word_importance: Array<{ word: string; score: number; fake_contribution: boolean }>;
  model_used: string;
  char_count: number;
  word_count: number;
};

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        {/* Hero */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/30 bg-primary/10 text-primary text-xs font-medium mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            AI-Powered Analysis
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold text-foreground leading-tight text-balance mb-4">
            Detect Fake News with
            <span className="text-primary"> Deep Learning</span>
          </h1>
          <p className="text-muted text-lg max-w-2xl mx-auto leading-relaxed text-pretty">
            Powered by LSTM, Bidirectional LSTM and BERT transformer models. Paste any news
            article to get an instant FAKE or REAL verdict with word-level explanations.
          </p>
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Input column */}
          <div className="lg:col-span-3 space-y-6">
            <PredictionForm
              onResult={setResult}
              onTextChange={setInputText}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
            <SampleNews onSelect={setInputText} />
          </div>

          {/* Result column */}
          <div className="lg:col-span-2 space-y-6">
            <ResultCard result={result} inputText={inputText} isLoading={isLoading} />
          </div>
        </div>

        {/* Model Comparison */}
        <div className="mt-12">
          <ModelComparison />
        </div>

        {/* How It Works */}
        <div className="mt-12">
          <HowItWorks />
        </div>
      </main>

      <footer className="border-t border-border mt-16 py-8 text-center text-muted text-sm">
        <p>
          FakeGuard &mdash; Deep Learning Fake News Detection &mdash; LSTM + BiLSTM + BERT
        </p>
        <p className="mt-1 text-xs opacity-60">
          For research and educational purposes. Always verify news from multiple credible sources.
        </p>
      </footer>
    </div>
  );
}
