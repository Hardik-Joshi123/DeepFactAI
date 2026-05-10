import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FakeGuard — AI-Powered Fake News Detector",
  description:
    "Detect fake news using state-of-the-art deep learning models including LSTM, Bidirectional LSTM, and BERT transformers. Get instant predictions with confidence scores and word-level explanations.",
  keywords: ["fake news", "NLP", "deep learning", "BERT", "LSTM", "AI", "fact check"],
  authors: [{ name: "FakeGuard AI" }],
  openGraph: {
    title: "FakeGuard — AI-Powered Fake News Detector",
    description: "Detect fake news using state-of-the-art deep learning models",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} bg-background`}>
      <body className="font-sans bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
