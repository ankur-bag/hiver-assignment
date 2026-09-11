import { ChatWindow } from '../components/ChatWindow';

export const metadata = {
  title: 'Hiver AI Customer Support Agent',
  description: 'Enterprise Multilingual Customer Support powered by RAG, Intent Classification, and Pinecone Vector Database',
};

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950">
      <ChatWindow />
    </main>
  );
}
