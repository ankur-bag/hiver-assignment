import React from 'react';

interface GlassPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  subtle?: boolean;
}

export const GlassPanel: React.FC<GlassPanelProps> = ({
  children,
  className = '',
  subtle = false,
  ...props
}) => {
  return (
    <div
      className={`rounded-2xl border transition-all duration-200 ${
        subtle
          ? 'bg-[#111116]/80 border-[rgba(255,255,255,0.06)]'
          : 'bg-[#15151C]/90 border-[rgba(255,255,255,0.08)] shadow-[0_8px_32px_rgba(0,0,0,0.36)] backdrop-blur-xl'
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
