import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "../components/Providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Jupbox - Music Control with Face Auth",
  description: "Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration",
  keywords: ["music", "spotify", "face authentication", "gesture control", "emotion detection"],
  authors: [{ name: "Jupbox Team" }],
  creator: "Jupbox",
  publisher: "Jupbox",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://jupbox.app'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: "Jupbox - Music Control with Face Auth",
    description: "Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration",
    url: 'https://jupbox.app',
    siteName: 'Jupbox',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Jupbox - Music Control with Face Auth',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: "Jupbox - Music Control with Face Auth",
    description: "Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration",
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  manifest: '/manifest.json',
  themeColor: '#8b5cf6',
  colorScheme: 'dark',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'Jupbox',
  },
  applicationName: 'Jupbox',
  referrer: 'origin-when-cross-origin',
  category: 'music',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="application-name" content="Jupbox" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Jupbox" />
        <meta name="description" content="Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="msapplication-config" content="/browserconfig.xml" />
        <meta name="msapplication-TileColor" content="#8b5cf6" />
        <meta name="msapplication-tap-highlight" content="no" />
        <meta name="theme-color" content="#8b5cf6" />

        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="manifest" href="/manifest.json" />
        <link rel="mask-icon" href="/safari-pinned-tab.svg" color="#8b5cf6" />
        <link rel="shortcut icon" href="/favicon.ico" />
        <link rel="msapplication-TileImage" content="/mstile-144x144.png" />

        <meta name="twitter:card" content="summary" />
        <meta name="twitter:url" content="https://jupbox.app" />
        <meta name="twitter:title" content="Jupbox - Music Control with Face Auth" />
        <meta name="twitter:description" content="Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration" />
        <meta name="twitter:image" content="https://jupbox.app/og-image.png" />
        <meta name="twitter:creator" content="@jupbox" />
        <meta name="twitter:site" content="@jupbox" />

        <meta property="og:type" content="website" />
        <meta property="og:title" content="Jupbox - Music Control with Face Auth" />
        <meta property="og:description" content="Your all-in-one music control system with gesture recognition, face authentication, and Spotify integration" />
        <meta property="og:site_name" content="Jupbox" />
        <meta property="og:url" content="https://jupbox.app" />
        <meta property="og:image" content="https://jupbox.app/og-image.png" />
      </head>
      <body className={inter.className}>
        <Providers>
          {children}
        </Providers>
        
        {/* PWA Service Worker Registration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js')
                    .then(function(registration) {
                      console.log('SW registered: ', registration);
                    })
                    .catch(function(registrationError) {
                      console.log('SW registration failed: ', registrationError);
                    });
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
