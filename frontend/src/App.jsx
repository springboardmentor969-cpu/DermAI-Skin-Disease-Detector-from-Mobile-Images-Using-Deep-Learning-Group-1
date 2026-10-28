import React, { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [statusMsg, setStatusMsg] = useState('Connected to DermAI Milestone 1 Core Workspace');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <header className="max-w-7xl mx-auto flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-teal-400">DermAI: Skin Disease Detector (Milestone 1)</h1>
          <p className="text-xs text-slate-400">Clinical Decision-Support & Screening Platform — API Endpoint: {API_BASE}</p>
        </div>
        <div className="flex gap-2 mt-3 sm:mt-0">
          {['upload', 'cases', 'eda', 'architecture'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium capitalize ${
                activeTab === tab ? 'bg-teal-500 text-slate-950' : 'bg-slate-800 text-slate-300'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>
      <main className="max-w-7xl mx-auto bg-slate-900 border border-slate-800 rounded-xl p-6">
        <p className="text-sm text-teal-300 mb-2">{statusMsg}</p>
        <p className="text-sm text-slate-300">
          See <code>DermAI_Milestone1_Platform.html</code> for the full zero-dependency interactive web app, or run <code>docker compose up --build</code> to connect this React client to FastAPI + PostgreSQL.
        </p>
      </main>
    </div>
  );
}
