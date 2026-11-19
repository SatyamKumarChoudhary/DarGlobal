/**
 * Dar Assist chat UI.
 *
 * Streams answers over SSE (POST + ReadableStream, since EventSource cannot
 * POST), renders sanitized markdown, and shows clickable source chips.
 */
(() => {
  "use strict";

  const API = {
    stream: "/api/v1/chat/stream",
    suggestions: "/api/v1/suggestions",
    stats: "/api/v1/stats",
    session: (id) => `/api/v1/sessions/${id}`,
  };

  const el = {
    chat: document.getElementById("chat"),
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("empty-state"),
    suggestions: document.getElementById("suggestions"),
    statsPill: document.getElementById("stats-pill"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("input"),
    send: document.getElementById("send"),
    newChat: document.getElementById("new-chat"),
  };

  let sessionId = sessionStorage.getItem("dar-assist-session") || null;
  let busy = false;

  /** Render markdown to sanitized HTML. */
  function renderMarkdown(text) {
    const raw = window.marked ? window.marked.parse(text) : text;
    return window.DOMPurify
      ? window.DOMPurify.sanitize(raw, { ADD_ATTR: ["target", "rel"] })
      : raw;
  }

  function scrollToBottom() {
    el.chat.scrollTop = el.chat.scrollHeight;
  }

  function hideEmptyState() {
    el.emptyState.hidden = true;
  }

  /** Append a message bubble; returns its content element. */
  function addMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (role === "user") {
      bubble.textContent = text;
    } else if (text) {
      bubble.innerHTML = renderMarkdown(text);
    }
    wrapper.appendChild(bubble);
    el.messages.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, bubble };
  }

  function addTypingIndicator(wrapper) {
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    wrapper.bubble.appendChild(typing);
    return typing;
  }

  function addSources(wrapper, sources) {
    if (!sources || sources.length === 0) return;
    const box = document.createElement("div");
    box.className = "sources";
    for (const source of sources) {
      const chip = document.createElement("a");
      chip.className = "source-chip";
      chip.href = source.url;
      chip.target = "_blank";
      chip.rel = "noopener noreferrer";
      const host = source.site === "darglobal" ? "darglobal.co.uk" : "wasalt.sa";
      chip.textContent = `[${source.index}] ${source.title} — ${host}`;
      chip.title = source.snippet || source.title;
      box.appendChild(chip);
    }
    wrapper.wrapper.appendChild(box);
    scrollToBottom();
  }

  function addErrorNote(wrapper, message) {
    const note = document.createElement("div");
    note.className = "bubble error-note";
    note.textContent = message;
    wrapper.wrapper.appendChild(note);
    scrollToBottom();
  }

  /** Parse an SSE byte stream, invoking onEvent(name, data) per event. */
  async function consumeSse(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventName = "message";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const payload = line.slice(5).trim();
          if (payload) onEvent(eventName, JSON.parse(payload));
        } else if (line === "") {
          eventName = "message";
        }
      }
    }
  }

  async function sendMessage(text) {
    if (busy || !text.trim()) return;
    busy = true;
    el.send.disabled = true;
    hideEmptyState();
    addMessage("user", text);

    const assistant = addMessage("assistant", "");
    const typing = addTypingIndicator(assistant);
    let answerText = "";

    try {
      const response = await fetch(API.stream, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        const message =
          (body && body.error && body.error.message) ||
          `Request failed (HTTP ${response.status})`;
        typing.remove();
        addErrorNote(assistant, message);
        return;
      }

      await consumeSse(response, (name, data) => {
        if (name === "meta") {
          sessionId = data.session_id;
          sessionStorage.setItem("dar-assist-session", sessionId);
        } else if (name === "token") {
          if (typing.parentNode) typing.remove();
          answerText += data.delta;
          assistant.bubble.innerHTML = renderMarkdown(answerText);
          scrollToBottom();
        } else if (name === "sources") {
          addSources(assistant, data.sources);
        } else if (name === "error") {
          if (typing.parentNode) typing.remove();
          addErrorNote(assistant, data.message);
        }
      });
    } catch (err) {
      if (typing.parentNode) typing.remove();
      addErrorNote(assistant, "Connection lost. Please try again.");
    } finally {
      if (typing.parentNode) typing.remove();
      busy = false;
      el.send.disabled = false;
      el.input.focus();
    }
  }

  /* ---------- Composer behaviour ---------- */
  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = el.input.value;
    el.input.value = "";
    autoResize();
    sendMessage(text);
  });

  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.form.requestSubmit();
    }
  });

  function autoResize() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 140)}px`;
  }
  el.input.addEventListener("input", autoResize);

  /* ---------- New chat ---------- */
  el.newChat.addEventListener("click", async () => {
    if (sessionId) {
      try {
        await fetch(API.session(sessionId), { method: "DELETE" });
      } catch (err) {
        /* best-effort cleanup */
      }
    }
    sessionId = null;
    sessionStorage.removeItem("dar-assist-session");
    el.messages.innerHTML = "";
    el.emptyState.hidden = false;
    el.input.focus();
  });

  /* ---------- Bootstrap: suggestions + stats ---------- */
  async function bootstrap() {
    try {
      const response = await fetch(API.suggestions);
      const data = await response.json();
      for (const question of data.suggestions || []) {
        const button = document.createElement("button");
        button.className = "suggestion";
        button.type = "button";
        button.textContent = question;
        button.addEventListener("click", () => sendMessage(question));
        el.suggestions.appendChild(button);
      }
    } catch (err) {
      /* suggestions are optional */
    }
    try {
      const response = await fetch(API.stats);
      const data = await response.json();
      if (data.n_documents) {
        el.statsPill.textContent = `${data.n_documents} pages indexed`;
        el.statsPill.hidden = false;
      }
    } catch (err) {
      /* stats are optional */
    }
  }

  bootstrap();
  el.input.focus();
})();
