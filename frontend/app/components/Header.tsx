"use client";

import { ShieldCheck, Github, BookOpen } from "lucide-react";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-primary/20 border border-primary/40 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4 text-primary" />
          </div>
          <span className="font-semibold text-foreground">FakeGuard</span>
          <span className="hidden sm:inline-block text-xs px-1.5 py-0.5 rounded bg-primary/20 text-primary font-mono">
            v1.0
          </span>
        </div>

        {/* Nav links */}
        <nav className="hidden sm:flex items-center gap-6 text-sm text-muted">
          <a href="#models" className="hover:text-foreground transition-colors">
            Models
          </a>
          <a href="#how-it-works" className="hover:text-foreground transition-colors">
            How it works
          </a>
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-muted hover:text-foreground transition-colors"
            aria-label="View on GitHub"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-border hover:border-primary/50 hover:text-primary transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5" />
            API Docs
          </a>
        </div>
      </div>
    </header>
  );
}
