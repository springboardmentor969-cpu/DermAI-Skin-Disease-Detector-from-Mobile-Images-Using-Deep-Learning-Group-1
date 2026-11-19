import React from 'react';

/**
 * DermAI Dermoscope Brand Mark
 * Circular dermoscope symbol with outer circle stroke, inner circle,
 * and 4 dots in a 2x2 grid representing a dermoscopy pigment network.
 */
export const DermoscopeLogo = ({ className = 'w-8 h-8', size = 32, color = '#0EA5E9' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`shrink-0 text-[#0EA5E9] dark:text-sky-400 ${className}`}
    aria-label="DermAI Dermoscope Logo"
  >
    {/* Outer circle stroke */}
    <circle
      cx="16"
      cy="16"
      r="14"
      stroke="currentColor"
      strokeWidth="2.25"
      className="stroke-[#0EA5E9] dark:stroke-sky-400"
    />
    {/* Inner circle */}
    <circle
      cx="16"
      cy="16"
      r="8.5"
      stroke="currentColor"
      strokeWidth="1.5"
      className="stroke-[#0EA5E9] dark:stroke-sky-400"
    />
    {/* 4 small dots in a 2x2 grid pattern inside */}
    <circle cx="13" cy="13" r="1.5" fill="currentColor" className="fill-[#0EA5E9] dark:fill-sky-400" />
    <circle cx="19" cy="13" r="1.5" fill="currentColor" className="fill-[#0EA5E9] dark:fill-sky-400" />
    <circle cx="13" cy="19" r="1.5" fill="currentColor" className="fill-[#0EA5E9] dark:fill-sky-400" />
    <circle cx="19" cy="19" r="1.5" fill="currentColor" className="fill-[#0EA5E9] dark:fill-sky-400" />
  </svg>
);

export default DermoscopeLogo;
