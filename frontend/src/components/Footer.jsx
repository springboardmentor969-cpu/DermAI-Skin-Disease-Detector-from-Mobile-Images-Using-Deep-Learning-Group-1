import React from 'react';
import { ShieldAlert } from 'lucide-react';

export const Footer = () => {
  return (
    <footer className="mt-auto border-t border-slate-200 bg-white/70 py-6 text-center">
      <div className="max-w-7xl mx-auto px-4 flex flex-col items-center justify-center space-y-2">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-xs font-medium">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
          <span>AI-assisted screening tool. Not a medical diagnosis.</span>
        </div>
        <p className="text-xs text-slate-500">
          DermAI Decision-Support System &copy; {new Date().getFullYear()} &bull; Strictly for educational screening and clinical research decision support.
        </p>
      </div>
    </footer>
  );
};
