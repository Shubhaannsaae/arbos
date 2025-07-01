import React from 'react';
import { useRouter } from 'next/router';
import { 
  ChartBarIcon, 
  CurrencyDollarIcon, 
  CogIcon,
  ShieldCheckIcon,
  HomeIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Arbitrage', href: '/arbitrage', icon: CurrencyDollarIcon },
  { name: 'Portfolio', href: '/portfolio', icon: ChartBarIcon },
  { name: 'AI Agents', href: '/agents', icon: BeakerIcon },
  { name: 'Security', href: '/security', icon: ShieldCheckIcon },
];

export const Sidebar: React.FC = () => {
  const router = useRouter();

  return (
    <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
      <div className="flex flex-col flex-grow pt-5 bg-gray-900 overflow-y-auto">
        <div className="flex items-center flex-shrink-0 px-4">
          <h1 className="text-white text-xl font-bold">ArbOS</h1>
        </div>
        
        <div className="mt-5 flex-1 flex flex-col">
          <nav className="flex-1 px-2 pb-4 space-y-1">
            {navigation.map((item) => {
              const isActive = router.pathname === item.href;
              return (
                <a
                  key={item.name}
                  href={item.href}
                  className={`${
                    isActive
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  } group flex items-center px-2 py-2 text-sm font-medium rounded-md`}
                >
                  <item.icon
                    className="mr-3 flex-shrink-0 h-6 w-6"
                    aria-hidden="true"
                  />
                  {item.name}
                </a>
              );
            })}
          </nav>
        </div>
      </div>
    </div>
  );
};
