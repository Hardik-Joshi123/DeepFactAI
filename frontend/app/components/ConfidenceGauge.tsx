"use client";

type Props = {
  fake: number;
  real: number;
};

function getConfidenceLevel(score: number): { label: string; color: string } {
  if (score >= 0.85) return { label: "High Confidence", color: "text-foreground" };
  if (score >= 0.70) return { label: "Moderate Confidence", color: "text-muted" };
  return { label: "Low Confidence", color: "text-muted/70" };
}

export function ConfidenceGauge({ fake, real }: Props) {
  const fakePct = Math.round(fake * 100);
  const realPct = Math.round(real * 100);
  const dominant = fake > real ? fake : real;
  const confidence = getConfidenceLevel(dominant);

  return (
    <div className="space-y-2">
      {/* Bar */}
      <div className="relative h-3 rounded-full bg-surface-2 overflow-hidden">
        {/* Fake portion (left side) */}
        <div
          className="absolute left-0 top-0 h-full bg-gradient-to-r from-danger to-danger/80 transition-all duration-700 ease-out"
          style={{ width: `${fakePct}%` }}
        />
        {/* Real portion (right side) */}
        <div
          className="absolute right-0 top-0 h-full bg-gradient-to-l from-success to-success/80 transition-all duration-700 ease-out"
          style={{ width: `${realPct}%` }}
        />
        {/* Center marker */}
        <div className="absolute left-1/2 top-0 w-0.5 h-full bg-background/50 -translate-x-1/2" />
      </div>

      {/* Labels */}
      <div className="flex justify-between items-center text-xs">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-danger inline-block shadow-sm shadow-danger/30" />
          <span className="text-muted">FAKE</span>
          <span className={`font-semibold tabular-nums ${fake > real ? 'text-danger' : 'text-foreground'}`}>
            {fakePct}%
          </span>
        </div>
        <div className={`text-xs ${confidence.color}`}>
          {confidence.label}
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`font-semibold tabular-nums ${real > fake ? 'text-success' : 'text-foreground'}`}>
            {realPct}%
          </span>
          <span className="text-muted">REAL</span>
          <span className="w-2.5 h-2.5 rounded-full bg-success inline-block shadow-sm shadow-success/30" />
        </div>
      </div>
    </div>
  );
}
