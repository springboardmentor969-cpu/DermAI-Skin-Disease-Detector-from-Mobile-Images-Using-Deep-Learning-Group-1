import React from 'react';
import { CheckCircle2, AlertTriangle, AlertOctagon } from 'lucide-react';

export const QualityBadge = ({ label, size = 'md' }) => {
  const normLabel = (label || '').toLowerCase();

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 space-x-1',
    md: 'text-xs px-2.5 py-1 space-x-1.5',
    lg: 'text-sm px-3.5 py-1.5 space-x-2',
  }[size] || 'text-xs px-2.5 py-1 space-x-1.5';

  if (normLabel === 'good') {
    return (
      <span
        className={`inline-flex items-center font-semibold rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 shadow-xs ${sizeClasses}`}
      >
        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
        <span>Good Quality</span>
      </span>
    );
  }

  if (normLabel === 'acceptable') {
    return (
      <span
        className={`inline-flex items-center font-semibold rounded-full bg-amber-50 text-amber-700 border border-amber-200 shadow-xs ${sizeClasses}`}
      >
        <AlertTriangle className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
        <span>Acceptable</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full bg-rose-50 text-rose-700 border border-rose-200 shadow-xs ${sizeClasses}`}
    >
      <AlertOctagon className="w-3.5 h-3.5 text-rose-600 flex-shrink-0" />
      <span>Poor Quality</span>
    </span>
  );
};
