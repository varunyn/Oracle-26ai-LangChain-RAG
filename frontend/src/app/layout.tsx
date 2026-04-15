import type { Metadata } from "next";

import { ConfigProvider } from "@/components/config-provider";
import { ToasterProvider } from "@/components/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getAppConfig } from "@/lib/config";
import "./globals.css";

export const metadata: Metadata = {
  title: "OCI Custom RAG Agent",
  description: "Chat with Oracle Cloud Infrastructure Generative AI using RAG",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const config = await getAppConfig();
  return (
    <html lang="en" className="h-full overflow-hidden">
      <body className="antialiased h-full overflow-hidden">
        <TooltipProvider>
          <ConfigProvider initialConfig={config}>
            <ToasterProvider>{children}</ToasterProvider>
          </ConfigProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
