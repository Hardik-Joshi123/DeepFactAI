"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, FlaskConical } from "lucide-react";

const SAMPLES = [
  {
    label: "FAKE",
    title: "Health Misinformation",
    text: "BREAKING: Scientists confirm that drinking bleach cures all known diseases according to a secret government report that has been suppressed by Big Pharma for decades. The whistleblower, who wishes to remain anonymous, claims that the FDA and CDC are deliberately hiding the cure to protect pharmaceutical profits. Share this before it gets taken down! Doctors hate this one weird trick!",
  },
  {
    label: "REAL",
    title: "Scientific Research",
    text: "Researchers at MIT have developed a new type of solar panel material that achieves a record-breaking efficiency of 47% in laboratory conditions, nearly doubling the performance of conventional silicon-based panels. The breakthrough, published in Nature Energy on Wednesday, uses a perovskite-silicon tandem cell architecture. According to lead researcher Dr. Sarah Chen, the technology could reduce solar energy costs by 40% within the next decade.",
  },
  {
    label: "FAKE",
    title: "Political Conspiracy",
    text: "EXPLOSIVE: Leaked documents reveal that George Soros and the deep state cabal paid thousands of protesters $1,500 each to stage riots across 50 American cities. Dominion Voting Systems, connected to servers in Venezuela, will flip millions of votes. The mainstream media is covering this up! Wake up America - they don't want you to know the truth! Share everywhere before this gets censored!",
  },
  {
    label: "REAL",
    title: "Economic News",
    text: "The Federal Reserve raised its benchmark interest rate by 25 basis points Wednesday, bringing it to a target range of 5.25% to 5.5%, the highest level in 22 years. Fed Chair Jerome Powell stated that while inflation has eased from its peak of 9.1% in June 2022, it remains above the central bank's 2% target. Officials signalled they may implement one more rate increase this year depending on incoming economic data.",
  },
  {
    label: "FAKE",
    title: "Celebrity Hoax",
    text: "SHOCKING: Famous actor found dead in hotel room - suicide suspected but MURDER covered up by Hollywood elites! Anonymous sources reveal terrifying details about what really happened. The mainstream media is refusing to report this! You won't believe the horrifying truth they're hiding from you. Share this before it gets deleted!!!",
  },
  {
    label: "REAL",
    title: "Technology News",
    text: "Apple reported quarterly revenue of $89.5 billion on Thursday, exceeding Wall Street expectations of $84.5 billion. iPhone sales grew 8% year-over-year to $43.8 billion, while Services revenue reached a record $21.2 billion. CEO Tim Cook announced the company will increase its dividend by 4% and authorized an additional $90 billion in share repurchases.",
  },
];

type Props = {
  onSelect: (text: string) => void;
};

export function SampleNews({ onSelect }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="glass-card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left hover:bg-surface-2/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-muted" />
          <span className="text-sm font-medium text-foreground">Sample News Articles</span>
          <span className="text-xs text-muted">({SAMPLES.length} examples)</span>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 text-muted" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted" />
        )}
      </button>

      {open && (
        <div className="border-t border-border divide-y divide-border/50">
          {SAMPLES.map((s, i) => (
            <button
              key={i}
              onClick={() => onSelect(s.text)}
              className="w-full text-left px-5 py-3.5 hover:bg-surface-2/60 transition-colors group"
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                    s.label === "FAKE"
                      ? "bg-danger/15 text-danger border border-danger/20"
                      : "bg-success/15 text-success border border-success/20"
                  }`}
                >
                  {s.label}
                </span>
                <span className="text-xs text-muted">{s.title}</span>
              </div>
              <p className="text-xs text-muted line-clamp-2 group-hover:text-foreground/80 transition-colors">
                {s.text}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
