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
    <html className="h-full overflow-hidden" lang="en">
      <body className="h-full overflow-hidden antialiased">
        <TooltipProvider>
          <ConfigProvider initialConfig={config}>
            <ToasterProvider>{children}</ToasterProvider>
          </ConfigProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
