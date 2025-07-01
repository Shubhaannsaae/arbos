import type { AppProps } from 'next/app';
import { useRouter } from 'next/router';
import Head from 'next/head';
import '../styles/globals.css';
import '../styles/components.css';
import { Header } from '../components/common/Header';
import { Sidebar } from '../components/common/Sidebar';

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter();
  const isHomePage = router.pathname === '/';

  return (
    <>
      <Head>
        <title>ArbOS - AI-Powered Cross-Chain Trading Platform</title>
        <meta name="description" content="Advanced arbitrage and portfolio management with Chainlink integration" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#3b82f6" />
      </Head>

      {isHomePage ? (
        <Component {...pageProps} />
      ) : (
        <div className="min-h-screen bg-gray-50">
          <Header />
          <div className="flex">
            <Sidebar />
            <main className="flex-1 md:ml-64 p-6">
              <Component {...pageProps} />
            </main>
          </div>
        </div>
      )}
    </>
  );
}
