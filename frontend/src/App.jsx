import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatView from "./components/ChatView";
import ReviewQueue from "./components/ReviewQueue";
import Dashboard from "./components/Dashboard";
import Training from "./components/Training";
import Articles from "./components/Articles";
import { loadConversations, newConversationId } from "./conversations";

export default function App() {
  const [page, setPage] = useState("chat");
  const [profile, setProfile] = useState("citoyen");
  const [conversationId, setConversationId] = useState(() => newConversationId());
  const [conversations, setConversations] = useState(() => loadConversations("citoyen"));

  // Chaque profil a son propre historique de conversations (un avocat et un
  // citoyen sur le même navigateur ne doivent pas se mélanger).
  useEffect(() => {
    setConversations(loadConversations(profile));
  }, [profile]);

  const startNewConversation = () => {
    setConversationId(newConversationId());
    setPage("chat");
  };

  const selectConversation = (id) => {
    setConversationId(id);
    setPage("chat");
  };

  const handleProfileChange = (nextProfile) => {
    setProfile(nextProfile);
    setConversationId(newConversationId()); // on ne mélange pas les fils de discussion entre profils
  };

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        onNavigate={setPage}
        profile={profile}
        onProfileChange={handleProfileChange}
        conversations={conversations}
        activeConversationId={conversationId}
        onNewConversation={startNewConversation}
        onSelectConversation={selectConversation}
      />
      <div className="main">
        {page === "chat" && (
          <ChatView
            key={`${profile}_${conversationId}`}
            profile={profile}
            conversationId={conversationId}
            onConversationsChange={setConversations}
          />
        )}
        {page === "articles" && <Articles />}
        {page === "review" && <ReviewQueue />}
        {page === "dashboard" && <Dashboard />}
        {page === "training" && <Training />}
      </div>
    </div>
  );
}