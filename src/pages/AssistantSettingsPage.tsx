import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  MessageSquare,
  Settings,
  Send,
  Sun,
  Moon,
  ShieldCheck,
  CheckCircle2,
  BookOpen,
} from 'lucide-react';
import {
  SUPPORTED_LANGUAGES,
  SupportedLanguageCode,
  setAppLanguage,
} from '../i18n';
import { SystemStatusResponse, agriApi } from '../api/client';
import { formatDateTime } from '../utils/format';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: string[];
  timestamp: string;
}

interface AssistantSettingsPageProps {
  systemStatus: SystemStatusResponse | null;
  darkMode: boolean;
  onToggleDarkMode: (dark: boolean) => void;
}

export const AssistantSettingsPage: React.FC<AssistantSettingsPageProps> = ({
  systemStatus,
  darkMode,
  onToggleDarkMode,
}) => {
  const { t, i18n } = useTranslation(['assistant', 'common']);
  const lang = i18n.language as SupportedLanguageCode;

  const [activeTab, setActiveTab] = useState<'assistant' | 'settings'>('assistant');
  const [inputMessage, setInputMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const handleSendQuestion = async (questionText?: string) => {
    const textToSend = (questionText ?? inputMessage).trim();
    if (!textToSend || sending) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      text: textToSend,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!questionText) setInputMessage('');
    setSending(true);
    setChatError(null);

    try {
      const res = await agriApi.askAssistant({
        message: textToSend,
        language: lang,
      });
      const botMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        text: res.reply,
        sources: res.sources,
        timestamp: res.timestamp,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Assistant request failed');
    } finally {
      setSending(false);
    }
  };

  const suggestedQuestions = [
    t('assistant:chat.q1'),
    t('assistant:chat.q2'),
    t('assistant:chat.q3'),
    t('assistant:chat.q4'),
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-[var(--agri-text)]">
          {t('assistant:title')}
        </h1>
        <p className="text-sm text-[var(--agri-muted)] mt-1 max-w-3xl">
          {t('assistant:subtitle')}
        </p>
      </div>

      {/* Two Simple Internal Tabs */}
      <div className="inline-flex items-center gap-1 p-1 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-lg">
        <button
          type="button"
          onClick={() => setActiveTab('assistant')}
          className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            activeTab === 'assistant'
              ? 'bg-[#173D2B] text-white'
              : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>{t('assistant:tabs.assistant')}</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('settings')}
          className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            activeTab === 'settings'
              ? 'bg-[#173D2B] text-white'
              : 'text-[var(--agri-muted)] hover:text-[var(--agri-text)]'
          }`}
        >
          <Settings className="w-3.5 h-3.5" />
          <span>{t('assistant:tabs.settings')}</span>
        </button>
      </div>

      {/* TAB 1: AI FARMING ASSISTANT */}
      {activeTab === 'assistant' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Chat Conversation Panel */}
          <div className="lg:col-span-8 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between min-h-[460px]">
            <div className="space-y-4 overflow-y-auto max-h-[420px] pr-1">
              {/* Welcome Block */}
              <div className="pb-4 border-b border-[var(--agri-border)]">
                <h2 className="text-base font-semibold text-[var(--agri-text)]">
                  {t('assistant:chat.welcomeTitle')}
                </h2>
                <p className="text-xs text-[var(--agri-muted)] mt-1">
                  {t('assistant:chat.welcomeDesc')}
                </p>
              </div>

              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`p-4 rounded-xl text-xs leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-[#173D2B] text-white ml-8'
                      : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border border-[var(--agri-border)] mr-4'
                  }`}
                >
                  <div className="whitespace-pre-line">{m.text}</div>

                  {m.sources && m.sources.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-[var(--agri-border)] text-[11px] text-[var(--agri-muted)]">
                      <div className="font-semibold flex items-center gap-1.5 text-[var(--agri-text)] mb-1">
                        <BookOpen className="w-3.5 h-3.5 text-[var(--agri-leaf)]" />
                        <span>{t('assistant:chat.sourcesLabel')}:</span>
                      </div>
                      <div>{m.sources.join(' · ')}</div>
                    </div>
                  )}

                  <div
                    className={`mt-2 text-[10px] font-mono-tabular ${
                      m.role === 'user' ? 'text-white/75' : 'text-[var(--agri-muted)]'
                    }`}
                  >
                    {formatDateTime(m.timestamp, lang)}
                  </div>
                </div>
              ))}

              {sending && (
                <div className="p-4 rounded-xl bg-[var(--agri-bg)] border border-[var(--agri-border)] text-xs text-[var(--agri-muted)] animate-pulse">
                  {t('assistant:chat.sending')}
                </div>
              )}

              {chatError && (
                <div className="p-3 rounded-lg border border-[var(--agri-red)] text-xs text-[var(--agri-red)]">
                  {chatError}
                </div>
              )}
            </div>

            {/* Message Input Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendQuestion();
              }}
              className="mt-5 pt-4 border-t border-[var(--agri-border)] flex items-center gap-3"
            >
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder={t('assistant:chat.inputPlaceholder')}
                className="flex-1 px-3.5 py-2.5 text-xs bg-[var(--agri-bg)] border border-[var(--agri-border)] rounded-lg text-[var(--agri-text)] placeholder:text-[var(--agri-muted)] focus:outline-none focus:border-[var(--agri-leaf)]"
              />
              <button
                type="submit"
                disabled={sending || !inputMessage.trim()}
                className="inline-flex items-center gap-2 px-4 py-2.5 text-xs font-semibold text-white bg-[#173D2B] hover:bg-[#23563D] rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:opacity-50"
              >
                <Send className="w-3.5 h-3.5" />
                <span>{t('assistant:chat.sendBtn')}</span>
              </button>
            </form>
          </div>

          {/* Suggested Agronomic Prompts */}
          <div className="lg:col-span-4 bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-3">
            <h3 className="text-sm font-semibold text-[var(--agri-text)]">
              {t('assistant:chat.suggestedTitle')}
            </h3>
            <div className="space-y-2.5">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  disabled={sending}
                  onClick={() => handleSendQuestion(q)}
                  className="w-full text-left p-3 text-xs text-[var(--agri-text)] bg-[var(--agri-bg)] hover:bg-[var(--agri-sage)] border border-[var(--agri-border)] rounded-lg transition-colors leading-relaxed cursor-pointer disabled:opacity-50"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: PLATFORM & LANGUAGE SETTINGS */}
      {activeTab === 'settings' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Language & Theme Preferences */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 space-y-6">
            {/* Language Switcher */}
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('assistant:settings.languageSection')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('assistant:settings.languageDesc')}
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 mt-4">
                {SUPPORTED_LANGUAGES.map((item) => {
                  const isCurrent = lang === item.code;
                  return (
                    <button
                      key={item.code}
                      type="button"
                      onClick={() => setAppLanguage(item.code)}
                      className={`px-3.5 py-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                        isCurrent
                          ? 'bg-[#173D2B] text-white border-[#173D2B]'
                          : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)] hover:bg-[var(--agri-sage)]'
                      }`}
                    >
                      <div className="text-sm font-semibold">{item.nativeLabel}</div>
                      <div
                        className={`text-[11px] ${
                          isCurrent ? 'text-white/80' : 'text-[var(--agri-muted)]'
                        }`}
                      >
                        {item.label} ({item.code})
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Theme Preference */}
            <div className="pt-5 border-t border-[var(--agri-border)]">
              <h2 className="text-base font-semibold text-[var(--agri-text)]">
                {t('assistant:settings.themeSection')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('assistant:settings.themeDesc')}
              </p>

              <div className="flex flex-wrap items-center gap-3 mt-3">
                <button
                  type="button"
                  onClick={() => onToggleDarkMode(false)}
                  className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border transition-colors cursor-pointer ${
                    !darkMode
                      ? 'bg-[#173D2B] text-white border-[#173D2B]'
                      : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)]'
                  }`}
                >
                  <Sun className="w-4 h-4" />
                  <span>{t('assistant:settings.lightMode')}</span>
                </button>
                <button
                  type="button"
                  onClick={() => onToggleDarkMode(true)}
                  className={`inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border transition-colors cursor-pointer ${
                    darkMode
                      ? 'bg-[#173D2B] text-white border-[#173D2B]'
                      : 'bg-[var(--agri-bg)] text-[var(--agri-text)] border-[var(--agri-border)]'
                  }`}
                >
                  <Moon className="w-4 h-4" />
                  <span>{t('assistant:settings.darkMode')}</span>
                </button>
              </div>
            </div>

            {/* Units Standard */}
            <div className="pt-5 border-t border-[var(--agri-border)]">
              <h2 className="text-sm font-semibold text-[var(--agri-text)]">
                {t('assistant:settings.unitsSection')}
              </h2>
              <p className="text-xs text-[var(--agri-muted)] mt-1">
                {t('assistant:settings.unitsValue')}
              </p>
            </div>
          </div>

          {/* System, IoT Broker & AI Service Status */}
          <div className="bg-[var(--agri-card)] border border-[var(--agri-border)] rounded-xl p-6 flex flex-col justify-between space-y-6">
            <div>
              <h2 className="text-base font-semibold text-[var(--agri-text)] mb-4">
                {t('assistant:settings.systemHealthSection')}
              </h2>

              <div className="divide-y divide-[var(--agri-border)] text-xs">
                <div className="py-3 first:pt-0 flex items-center justify-between gap-4">
                  <span className="text-[var(--agri-muted)]">
                    {t('assistant:settings.backendStatus')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-leaf)] inline-flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    {systemStatus?.backendVersion ?? t('common:status.online')}
                  </span>
                </div>

                <div className="py-3 flex items-center justify-between gap-4">
                  <span className="text-[var(--agri-muted)]">
                    {t('assistant:settings.mqttStatus')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-leaf)]">
                    {systemStatus?.mqttBroker ?? 'connected'} (TLS 1.3)
                  </span>
                </div>

                <div className="py-3 flex items-center justify-between gap-4">
                  <span className="text-[var(--agri-muted)]">
                    {t('assistant:settings.edgeStatus')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {systemStatus?.edgeGateway ?? 'RPI-GW-01'}
                  </span>
                </div>

                <div className="py-3 last:pb-0 flex items-center justify-between gap-4">
                  <span className="text-[var(--agri-muted)]">
                    {t('assistant:settings.aiStatus')}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[var(--agri-text)]">
                    {systemStatus?.aiService ?? 'Ready'}
                  </span>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-[var(--agri-bg)] border border-[var(--agri-border)] flex items-start gap-3 text-xs text-[var(--agri-muted)]">
              <ShieldCheck className="w-4 h-4 text-[var(--agri-leaf)] shrink-0 mt-0.5" />
              <span>{t('assistant:settings.securityNotice')}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
