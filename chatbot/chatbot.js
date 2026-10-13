/*
 * FlowSight 문서 검색 챗봇 위젯
 * 화면 오른쪽 아래 동그란 버튼 → 누르면 대화창이 열린다.
 * 질문은 Supabase Edge Function(chat)으로 보내고, 그쪽에서 문서 검색(RAG) + Gemini 답변을 만든다.
 *
 * 사용법: <script src="chatbot/chatbot.js" defer></script>
 * 대화 내용은 이 탭의 메모리에만 있고 저장하거나 서버 DB에 남기지 않는다.
 */
(function () {
  "use strict";

  // publishable 키는 브라우저 공개용 키다 (테이블은 RLS로 막혀 있어 이 키로 문서를 직접 읽을 수 없음)
  var ENDPOINT = "https://sjbyofuuuhrzpkgrwoax.supabase.co/functions/v1/chat";
  var PUBLISHABLE_KEY = "sb_publishable_N0HZi8381E1VBgnuTpmvOw_IEX08Ixe";

  var MAX_QUESTION = 500;
  var SUGGESTIONS = [
    "FlowSight는 어떻게 인원을 세나요?",
    "영상은 어디에 저장되나요?",
    "어떤 비전 모델을 쓰나요?",
    "전처리 단계를 알려주세요",
  ];
  // 출처 표시용 짧은 문서 이름 (없으면 DB의 문서 제목을 그대로 씀)
  var SOURCE_NAMES = {
    "index.html": "홈페이지",
    "docs.html": "비전 모델 비교",
    "README.md": "README",
    "PRD/PRD.md": "기획서(PRD)",
    "report/preprocess.md": "전처리 설계",
    "report/model_plan.md": "모델 계획",
    "report/vision_models.md": "비전 모델 조사",
    "report/checklist.md": "현장 조사 가이드",
  };

  var CSS = [
    ".fs-chat{--fs-navy:#141a32;--fs-surface:#ffffff;--fs-bg:#f8f9ff;--fs-soft:#eff4ff;--fs-line:#d3e4fe;--fs-text:#0b1c30;--fs-muted:#46464d;--fs-accent:#10b981;",
    "position:fixed;right:24px;bottom:24px;z-index:9999;font-family:'Hanken Grotesk',system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif;color:var(--fs-text)}",
    ".fs-chat *{box-sizing:border-box}",
    ".fs-chat-fab{width:60px;height:60px;border-radius:50%;border:0;cursor:pointer;background:var(--fs-navy);color:#fff;display:flex;align-items:center;justify-content:center;",
    "box-shadow:0 10px 30px -8px rgba(20,26,50,.55);transition:transform .2s ease,box-shadow .2s ease}",
    ".fs-chat-fab:hover{transform:scale(1.06);box-shadow:0 14px 36px -8px rgba(20,26,50,.65)}",
    ".fs-chat-fab:focus-visible,.fs-chat button:focus-visible,.fs-chat textarea:focus-visible{outline:3px solid #497cff;outline-offset:2px}",
    ".fs-chat-fab svg{width:28px;height:28px}",
    ".fs-chat-fab .fs-i-close{display:none}",
    ".fs-chat.is-open .fs-chat-fab .fs-i-chat{display:none}",
    ".fs-chat.is-open .fs-chat-fab .fs-i-close{display:block}",
    ".fs-chat-dot{position:absolute;top:4px;right:4px;width:12px;height:12px;border-radius:50%;background:var(--fs-accent);border:2px solid #fff}",
    ".fs-chat.is-open .fs-chat-dot{display:none}",
    ".fs-chat-panel{position:absolute;right:0;bottom:76px;width:380px;height:560px;max-height:calc(100vh - 120px);background:var(--fs-surface);border-radius:24px;",
    "box-shadow:0 24px 64px -16px rgba(11,28,48,.45);display:flex;flex-direction:column;overflow:hidden;",
    "opacity:0;transform:translateY(12px) scale(.98);transform-origin:bottom right;pointer-events:none;visibility:hidden;transition:opacity .2s ease,transform .2s ease,visibility 0s .2s}",
    ".fs-chat.is-open .fs-chat-panel{opacity:1;transform:none;pointer-events:auto;visibility:visible;transition:opacity .2s ease,transform .2s ease}",
    ".fs-chat-head{background:var(--fs-navy);color:#fff;padding:16px 18px;display:flex;align-items:center;gap:12px}",
    ".fs-chat-avatar{width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,.12);display:flex;align-items:center;justify-content:center;flex-shrink:0}",
    ".fs-chat-avatar svg{width:20px;height:20px}",
    ".fs-chat-title{font-family:Manrope,inherit;font-weight:700;font-size:16px;line-height:20px}",
    ".fs-chat-sub{font-size:12px;color:#dce9ff;display:flex;align-items:center;gap:6px}",
    ".fs-chat-sub::before{content:'';width:7px;height:7px;border-radius:50%;background:var(--fs-accent)}",
    ".fs-chat-x{margin-left:auto;background:transparent;border:0;color:#dce9ff;cursor:pointer;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center}",
    ".fs-chat-x:hover{background:rgba(255,255,255,.12);color:#fff}",
    ".fs-chat-x svg{width:20px;height:20px}",
    ".fs-chat-log{flex:1;overflow-y:auto;padding:18px 16px;background:var(--fs-bg);display:flex;flex-direction:column;gap:12px;scrollbar-width:thin}",
    ".fs-chat-log::-webkit-scrollbar{display:block;width:6px}",
    ".fs-chat-log::-webkit-scrollbar-thumb{background:var(--fs-line);border-radius:3px}",
    ".fs-msg{max-width:86%;padding:10px 14px;border-radius:18px;font-size:14px;line-height:22px;white-space:pre-wrap;word-break:keep-all;overflow-wrap:anywhere}",
    ".fs-msg.user{align-self:flex-end;background:var(--fs-navy);color:#fff;border-bottom-right-radius:6px}",
    ".fs-msg.bot{align-self:flex-start;background:var(--fs-surface);border-bottom-left-radius:6px;box-shadow:0 1px 3px rgba(11,28,48,.08)}",
    ".fs-msg.error{background:#ffdad6;color:#93000a}",
    ".fs-msg.notice{background:#fffbeb;color:#6b4e00;box-shadow:inset 0 0 0 1px #f5d98b}",
    ".fs-msg code{background:var(--fs-soft);padding:1px 5px;border-radius:5px;font-size:13px}",
    ".fs-sources{margin-top:8px;padding-top:6px;border-top:1px solid var(--fs-soft);font-size:11.5px;line-height:16px;color:#76767e;white-space:normal}",
    ".fs-suggest{display:flex;flex-wrap:wrap;gap:8px}",
    ".fs-suggest button{border:1px solid var(--fs-line);background:var(--fs-surface);color:var(--fs-text);font:inherit;font-size:13px;padding:7px 12px;border-radius:999px;cursor:pointer;text-align:left}",
    ".fs-suggest button:hover{background:var(--fs-soft)}",
    ".fs-typing{display:inline-flex;gap:4px;align-items:center;height:22px}",
    ".fs-typing span{width:7px;height:7px;border-radius:50%;background:#76767e;animation:fs-bounce 1.2s infinite ease-in-out}",
    ".fs-typing span:nth-child(2){animation-delay:.15s}.fs-typing span:nth-child(3){animation-delay:.3s}",
    "@keyframes fs-bounce{0%,80%,100%{opacity:.3;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}",
    ".fs-chat-form{display:flex;align-items:flex-end;gap:8px;padding:12px;border-top:1px solid var(--fs-soft);background:var(--fs-surface)}",
    ".fs-chat-form textarea{flex:1;resize:none;border:1px solid var(--fs-line);border-radius:16px;padding:10px 14px;font:inherit;font-size:14px;line-height:20px;max-height:120px;color:var(--fs-text);background:var(--fs-surface)}",
    ".fs-chat-form textarea:focus{border-color:#497cff;outline:none}",
    ".fs-chat-send{width:42px;height:42px;border-radius:50%;border:0;background:var(--fs-navy);color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0}",
    ".fs-chat-send:disabled{opacity:.4;cursor:not-allowed}",
    ".fs-chat-send svg{width:20px;height:20px}",
    ".fs-chat-note{font-size:11px;color:var(--fs-muted);text-align:center;padding:0 12px 10px;background:var(--fs-surface)}",
    "@media (max-width:480px){.fs-chat{right:16px;bottom:16px}",
    ".fs-chat-panel{position:fixed;left:16px;right:16px;bottom:88px;width:auto;height:auto;top:96px;max-height:none}}",
    "@media (prefers-reduced-motion:reduce){.fs-chat-panel,.fs-chat-fab{transition:none}.fs-typing span{animation:none}}",
  ].join("");

  var ICON = {
    chat: '<svg class="fs-i-chat" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h.01M12 11h.01M15.5 11h.01" stroke-width="2.6"/></svg>',
    close: '<svg class="fs-i-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    bot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01"/></svg>',
    send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
  };

  var history = []; // {role:'user'|'assistant', content}
  var busy = false;

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // 답변용 최소 마크다운: **굵게**, `코드`, 목록 기호만 처리 (HTML은 모두 이스케이프)
  function renderAnswer(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/^\s*[-*]\s+/gm, "• ");
  }

  function build() {
    var style = el("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    var root = el("div", "fs-chat");
    root.innerHTML =
      '<div class="fs-chat-panel" role="dialog" aria-modal="false" aria-labelledby="fs-chat-title" id="fs-chat-panel">' +
      '<div class="fs-chat-head"><div class="fs-chat-avatar">' + ICON.bot + "</div>" +
      '<div><div class="fs-chat-title" id="fs-chat-title">FlowSight 도우미</div><div class="fs-chat-sub">프로젝트 문서에서 찾아 답해요</div></div>' +
      '<button type="button" class="fs-chat-x" aria-label="챗봇 닫기">' + ICON.close.replace('class="fs-i-close"', "") + "</button></div>" +
      '<div class="fs-chat-log" aria-live="polite"></div>' +
      '<form class="fs-chat-form"><label for="fs-chat-input" style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)">질문 입력</label>' +
      '<textarea id="fs-chat-input" rows="1" maxlength="' + MAX_QUESTION + '" placeholder="궁금한 점을 물어보세요"></textarea>' +
      '<button type="submit" class="fs-chat-send" aria-label="보내기" disabled>' + ICON.send + "</button></form>" +
      '<div class="fs-chat-note">AI 답변은 틀릴 수 있어요. 중요한 내용은 원문 문서를 확인하세요.</div>' +
      "</div>" +
      '<button type="button" class="fs-chat-fab" aria-label="챗봇 열기" aria-expanded="false" aria-controls="fs-chat-panel">' +
      ICON.chat + ICON.close + '<span class="fs-chat-dot" aria-hidden="true"></span></button>';
    document.body.appendChild(root);

    var fab = root.querySelector(".fs-chat-fab");
    var closeBtn = root.querySelector(".fs-chat-x");
    var log = root.querySelector(".fs-chat-log");
    var form = root.querySelector(".fs-chat-form");
    var input = root.querySelector("textarea");
    var send = root.querySelector(".fs-chat-send");

    function setOpen(open) {
      root.classList.toggle("is-open", open);
      fab.setAttribute("aria-expanded", String(open));
      fab.setAttribute("aria-label", open ? "챗봇 닫기" : "챗봇 열기");
      if (open) {
        if (!log.childElementCount) greet();
        setTimeout(function () { input.focus(); }, 50);
      } else {
        fab.focus();
      }
    }

    function scrollDown() {
      log.scrollTop = log.scrollHeight;
    }

    function addMsg(role, html) {
      var m = el("div", "fs-msg " + role, html);
      log.appendChild(m);
      scrollDown();
      return m;
    }

    function greet() {
      addMsg("bot", "안녕하세요! FlowSight 프로젝트 문서(기획서, 전처리 설계, 모델 계획 등)를 바탕으로 답해 드려요. 무엇이 궁금하세요?");
      showSuggestions();
    }

    function showSuggestions() {
      var old = log.querySelector(".fs-suggest");
      if (old) old.remove();
      var box = el("div", "fs-suggest");
      SUGGESTIONS.forEach(function (q) {
        var b = el("button");
        b.type = "button";
        b.textContent = q;
        b.addEventListener("click", function () {
          box.remove();
          ask(q);
        });
        box.appendChild(b);
      });
      log.appendChild(box);
      scrollDown();
    }

    // 답변 아래에 참고한 문서 이름만 작게 표시한다 (링크 없음: 누르면 페이지가 바뀌어 대화가 사라지므로)
    function renderSources(msgEl, sources) {
      if (!sources || !sources.length) return;
      var names = [];
      sources.forEach(function (s) {
        var name = SOURCE_NAMES[s.source] || s.title;
        if (names.indexOf(name) < 0) names.push(name);
      });
      var wrap = el("div", "fs-sources");
      wrap.textContent = "참고한 문서 · " + names.join(", ");
      msgEl.appendChild(wrap);
      scrollDown();
    }

    function ask(question) {
      question = question.trim();
      if (!question || busy) return;
      busy = true;
      send.disabled = true;
      var sug = log.querySelector(".fs-suggest");
      if (sug) sug.remove();

      addMsg("user", escapeHtml(question));
      var pending = addMsg("bot", '<span class="fs-typing" aria-label="답변 작성 중"><span></span><span></span><span></span></span>');

      fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json", apikey: PUBLISHABLE_KEY },
        body: JSON.stringify({ question: question, history: history.slice(-6) }),
      })
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (data) {
            if (!res.ok || !data.answer) throw new Error(data.error || "답변을 가져오지 못했어요.");
            return data;
          });
        })
        .then(function (data) {
          pending.innerHTML = renderAnswer(data.answer);
          if (data.blocked) {
            // 주제 밖 질문: 안내만 보여주고 대화 기록에는 남기지 않는다 (다음 검색을 흐리지 않도록)
            pending.classList.add("notice");
            showSuggestions();
            return;
          }
          renderSources(pending, data.sources);
          history.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
        })
        .catch(function (err) {
          pending.classList.add("error");
          pending.textContent = err && err.message && err.message !== "Failed to fetch"
            ? err.message
            : "네트워크 오류가 발생했어요. 잠시 후 다시 시도해 주세요.";
        })
        .then(function () {
          busy = false;
          send.disabled = !input.value.trim();
          scrollDown();
        });
    }

    function autosize() {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 120) + "px";
      send.disabled = busy || !input.value.trim();
    }

    fab.addEventListener("click", function () {
      setOpen(!root.classList.contains("is-open"));
    });
    closeBtn.addEventListener("click", function () { setOpen(false); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && root.classList.contains("is-open")) setOpen(false);
    });
    input.addEventListener("input", autosize);
    input.addEventListener("keydown", function (e) {
      // Enter 전송, Shift+Enter 줄바꿈 (한글 조합 중 Enter는 무시)
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit"));
      }
    });
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (busy || !input.value.trim()) return;
      var q = input.value;
      input.value = "";
      autosize();
      ask(q);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
  else build();
})();
