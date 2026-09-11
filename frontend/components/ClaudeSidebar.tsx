'use client';

import React from 'react';

interface SidebarProps {
  isOpen: boolean;
  onClose?: () => void;
  onNewChat: () => void;
  activeSessionId?: string;
}

const RECENT_CHATS = [
  'Internship termination response',
  'Building an AI customer support agent for Hiver',
  'Indian ports and beaches CSV files',
  'Software engineer test assignment submission',
  'Highest executive positions in tech companies',
  'Converting dataset to CSV format',
  'Code review assessment under time pressure',
  'Browser location access and PIN security',
  'Redis vs Upstash Redis differences',
  'Computer revolution presentation outline',
  'Cyclone TARANG database verification',
  'Marine intelligence datasets for vector database',
  'Creating a moving SVG animation',
  'Portfolio of hackathon wins and full-stack projects',
];

export const ClaudeSidebar: React.FC<SidebarProps> = ({
  isOpen,
  onNewChat,
}) => {
  return (
    <aside
      className={`fixed lg:static top-0 left-0 bottom-0 z-40 w-64 bg-[#171717] border-r border-[rgba(255,255,255,0.06)] flex flex-col transition-transform duration-200 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
      }`}
    >
      {/* Top Section */}
      <div className="p-3.5 pb-2">
        <div className="flex items-center justify-between px-2 py-1 mb-2">
          <span className="font-serif-claude text-xl font-normal tracking-wide text-[#ECECEC]">
            Claude
          </span>
          <div className="flex items-center gap-1 text-[#8F8F8F]">
            <button
              onClick={onNewChat}
              title="Sidebar icon"
              className="p-1 rounded-md hover:bg-white/[0.06] hover:text-[#ECECEC] transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="1.7" />
                <path d="M9 3v18" strokeWidth="1.7" />
              </svg>
            </button>
            <button
              title="Search chats"
              className="p-1 rounded-md hover:bg-white/[0.06] hover:text-[#ECECEC] transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="7" strokeWidth="1.7" />
                <path d="M21 21l-4.35-4.35" strokeWidth="1.7" />
              </svg>
            </button>
          </div>
        </div>

        {/* Start new chat button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg bg-[#242424] hover:bg-[#2B2B2B] text-xs font-medium text-[#ECECEC] transition-colors border border-[rgba(255,255,255,0.04)] shadow-xs"
        >
          <span className="text-base leading-none text-[#9E9E9E]">+</span>
          <span>New chat</span>
        </button>

        {/* Main navigation list */}
        <nav className="mt-3 space-y-0.5 text-xs text-[#9E9E9E]">
          <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/[0.04] hover:text-[#ECECEC] cursor-pointer transition-colors">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
            <span>Projects</span>
          </div>
          <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/[0.04] hover:text-[#ECECEC] cursor-pointer transition-colors">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span>Artifacts</span>
          </div>
          <div className="flex items-center justify-between px-3 py-1.5 rounded-lg hover:bg-white/[0.04] hover:text-[#ECECEC] cursor-pointer transition-colors">
            <div className="flex items-center gap-2.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              <span>Code</span>
            </div>
            <span className="text-[10px] text-[#A1A1AA] bg-[#262626] px-1.5 py-0.2 rounded border border-white/[0.06]">
              Upgrade
            </span>
          </div>
          <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/[0.04] hover:text-[#ECECEC] cursor-pointer transition-colors">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            <span>Customize</span>
          </div>
        </nav>
      </div>

      {/* Recents list */}
      <div className="flex-1 overflow-y-auto px-3.5 py-2 space-y-4">
        <div>
          <div className="px-3 pb-1 text-[11px] font-medium text-[#666666] tracking-tight">
            Pinned
          </div>
          <div className="text-xs text-[#9E9E9E] px-3 py-1.5 rounded-md hover:bg-white/[0.04] hover:text-[#ECECEC] cursor-pointer truncate">
            Software engineer intern application with Hiver
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between px-3 pb-1 text-[11px] font-medium text-[#666666] tracking-tight">
            <span>Chats and tasks</span>
            <svg className="w-3 h-3 text-[#666666]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
            </svg>
          </div>
          <div className="space-y-0.5">
            {RECENT_CHATS.map((title, i) => (
              <div
                key={i}
                className="group flex items-center justify-between text-xs text-[#9E9E9E] hover:text-[#ECECEC] px-3 py-1.5 rounded-md hover:bg-white/[0.04] cursor-pointer transition-colors"
              >
                <span className="truncate pr-2">{title}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom User Profile Section */}
      <div className="p-3 border-t border-[rgba(255,255,255,0.06)] bg-[#171717]">
        <div className="flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-white/[0.04] cursor-pointer transition-colors">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-full bg-[#2C2C2E] border border-white/[0.08] flex items-center justify-center text-[11px] font-medium text-[#ECECEC]">
              H
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-medium text-[#ECECEC] truncate max-w-[120px]">
                Hiver Support
              </span>
              <span className="text-[10px] text-[#666666]">Pro Team</span>
            </div>
          </div>
          <svg className="w-4 h-4 text-[#666666]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
    </aside>
  );
};
