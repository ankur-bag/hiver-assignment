'use client';

import React from 'react';
import { FiPlus, FiSettings, FiMessageSquare, FiSidebar, FiTrash2 } from 'react-icons/fi';
import { ConversationItem, ServiceHealth } from '../types/chat';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  onOpenSettings: () => void;
  health: ServiceHealth;
  sessionId: string;
  conversations?: ConversationItem[];
  onSelectConversation?: (id: string) => void;
  onDeleteConversation?: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onToggle,
  onNewChat,
  onOpenSettings,
  health,
  sessionId,
  conversations = [],
  onSelectConversation,
  onDeleteConversation,
}) => {
  return (
    <>
      {/* Mobile overlay backdrop */}
      {isOpen && (
        <div
          onClick={onToggle}
          className="fixed inset-0 z-40 bg-black/50 md:hidden backdrop-blur-xs transition-opacity cursor-pointer"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col w-[230px] shrink-0 bg-[#1E1D1B] border-r border-white/[0.06] transition-transform duration-200 ease-in-out select-none ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:hidden'
        }`}
      >
        {/* Top brand header */}
        <div className="h-14 px-4 flex items-center justify-between border-b border-white/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-md bg-amber-500/90 text-[#171615] flex items-center justify-center font-bold text-xs shadow-xs">
              H
            </div>
            <span className="text-[13px] font-medium tracking-tight text-[#EDEDEC]">
              Hiver AI Support
            </span>
          </div>

          <button
            onClick={onToggle}
            className="p-1.5 rounded-md hover:bg-white/[0.06] text-[#8C8B88] hover:text-[#EDEDEC] transition-colors cursor-pointer"
            title="Collapse sidebar"
            aria-label="Collapse sidebar"
          >
            <FiSidebar className="w-4 h-4" />
          </button>
        </div>

        {/* Action Button: New Conversation */}
        <div className="p-3">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-[13px] font-normal text-[#EDEDEC] transition-colors cursor-pointer"
          >
            <FiPlus className="w-4 h-4 text-amber-400" />
            <span>New conversation</span>
          </button>
        </div>

        {/* Conversations List Section */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
          <div className="px-2 py-1 text-[11px] font-medium tracking-wide text-[#8C8B88]">
            Conversations
          </div>

          {conversations.length > 0 ? (
            conversations.map((conv) => {
              const isActive = conv.id === sessionId;
              return (
                <div
                  key={conv.id}
                  onClick={() => onSelectConversation?.(conv.id)}
                  className={`group flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-[12.5px] transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-white/[0.08] text-[#EDEDEC] font-medium border border-white/[0.08]'
                      : 'text-[#8C8B88] hover:text-[#EDEDEC] hover:bg-white/[0.04]'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FiMessageSquare
                      className={`w-3.5 h-3.5 shrink-0 ${
                        isActive ? 'text-amber-400' : 'text-[#8C8B88] group-hover:text-[#EDEDEC]'
                      }`}
                    />
                    <span className="truncate">{conv.title || 'Conversation'}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation?.(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-400 text-[#8C8B88] transition-opacity cursor-pointer"
                    title="Delete conversation"
                    aria-label="Delete conversation"
                  >
                    <FiTrash2 className="w-3 h-3" />
                  </button>
                </div>
              );
            })
          ) : (
            <div className="px-2 py-3 text-[12px] text-[#63625F]">
              No previous conversations
            </div>
          )}
        </div>

        {/* Bottom footer: Status and Settings */}
        <div className="p-3 border-t border-white/[0.06] space-y-2">
          {/* Status Indicator */}
          <div className="flex items-center justify-between px-2 py-1 text-[11.5px] text-[#8C8B88]">
            <span className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  health.status === 'healthy'
                    ? 'bg-emerald-400'
                    : health.status === 'degraded'
                    ? 'bg-amber-400'
                    : 'bg-rose-400'
                }`}
              />
              <span>System status</span>
            </span>
            <span className="text-[11px] text-[#63625F]">
              {health.status === 'healthy' ? 'Operational' : health.status}
            </span>
          </div>

          {/* Settings Trigger */}
          <button
            onClick={onOpenSettings}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg hover:bg-white/[0.04] text-[12.5px] text-[#8C8B88] hover:text-[#EDEDEC] transition-colors cursor-pointer"
          >
            <FiSettings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>
        </div>
      </aside>
    </>
  );
};
