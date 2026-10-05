import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Topic Desk | News classification",
  description: "Classify news with a trained model and explore its measured performance.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
