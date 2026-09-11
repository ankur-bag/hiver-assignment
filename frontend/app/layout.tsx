import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Hiver AI Support Agent — AI Operations Console",
  description: "Enterprise Multilingual Customer Support powered by Hybrid Intent Classification, Pinecone Vector Retrieval, and Gemini RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-[#0B0B0F] text-[#F3F4F6] selection:bg-amber-500/20 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
