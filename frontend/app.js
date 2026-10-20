"use strict";

const startButton = document.querySelector("#start-button");
const startFormButton = document.querySelector("#start-form-button");
const introCard = document.querySelector(".intro-card");
const formPanel = document.querySelector("#form-panel");
const screeningForm = document.querySelector("#screening-form");
const formFields = document.querySelector("#form-fields");
const formSubmitButton = document.querySelector("#form-submit-button");
const backToIntroButton = document.querySelector("#back-to-intro-button");
const formStatusMessage = document.querySelector("#form-status-message");
const formResultPanel = document.querySelector("#form-result");
const conversationPanel = document.querySelector("#conversation-panel");
const companion = document.querySelector(".sneha-companion");
const botCaption = document.querySelector("#bot-caption");
const voiceToggle = document.querySelector("#voice-toggle");
const messages = document.querySelector("#messages");
const replyForm = document.querySelector("#reply-form");
const replyInput = document.querySelector("#reply-input");
const sendButton = document.querySelector("#send-button");
const progressBar = document.querySelector("#progress-bar");
const progressLabel = document.querySelector("#progress-label");
const stageLabel = document.querySelector("#stage-label");
const statusMessage = document.querySelector("#status-message");
const resultPanel = document.querySelector("#screening-result");

let sessionId = null;
let requestPending = false;
let currentResponse = {};
let formFieldDefinitions = [];
let formSchemaLoaded = false;
let companionTimeout = null;
let voiceEnabled = true;
let activeUtterance = null;
let femaleVoiceFallbackReported = false;
let activeAudio = null;
let activeAudioUrl = null;
let voiceRequestId = 0;
const browserSpeechSupported =
  "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
const voicePlaybackSupported =
  browserSpeechSupported || typeof window.Audio === "function";

function findFemaleEnglishVoice() {
  const femaleNamePattern =
    /\b(female|woman|zira|heera|samantha|victoria|susan|aria|jenny|ava|emma|sonia|kendra|joanna|ivy|tessa|kate|lucy|serena|hazel|catherine|salli|nicole|olivia|ananya|eva)\b/i;
  const englishVoices = window.speechSynthesis
    .getVoices()
    .filter((voice) => voice.lang.toLowerCase().startsWith("en"));
  const femaleVoices = englishVoices.filter((voice) =>
    femaleNamePattern.test(`${voice.name} ${voice.voiceURI}`),
  );
  const languagePriority = ["en-in", "en-gb", "en-us"];

  femaleVoices.sort((left, right) => {
    const leftLanguage = languagePriority.indexOf(left.lang.toLowerCase());
    const rightLanguage = languagePriority.indexOf(right.lang.toLowerCase());
    const leftPriority = leftLanguage === -1 ? languagePriority.length : leftLanguage;
    const rightPriority =
      rightLanguage === -1 ? languagePriority.length : rightLanguage;
    return leftPriority - rightPriority;
  });

  return {
    voice: femaleVoices[0] || null,
    hasEnglishVoices: englishVoices.length > 0,
  };
}

function setCompanionState(state, duration = 2400) {
  if (!companion) return;

  window.clearTimeout(companionTimeout);
  companion.classList.remove("is-thinking", "is-speaking");
  if (state === "thinking") {
    companion.classList.add("is-thinking");
    botCaption.textContent = "Thinking...";
    return;
  }
  if (state === "speaking") {
    companion.classList.add("is-speaking");
    botCaption.textContent = "Sneha is responding";
    if (duration !== null) {
      companionTimeout = window.setTimeout(() => {
        companion.classList.remove("is-speaking");
        botCaption.textContent = "Here with you";
      }, duration);
    }
    return;
  }

  botCaption.textContent = "Here with you";
}

function stopVoicePlayback() {
  voiceRequestId += 1;
  if (browserSpeechSupported) {
    activeUtterance = null;
    window.speechSynthesis.cancel();
  }
  if (activeAudio) {
    activeAudio.pause();
    activeAudio = null;
  }
  if (activeAudioUrl) {
    URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = null;
  }
  setCompanionState("idle");
}

function speakWithBrowser(content, status) {
  if (!browserSpeechSupported) {
    statusMessage.textContent = status;
    setCompanionState("idle");
    return;
  }
  try {
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = document.documentElement.lang || "en";
    const voiceChoice = findFemaleEnglishVoice();
    if (voiceChoice.voice) {
      utterance.voice = voiceChoice.voice;
    } else {
      utterance.pitch = 1.18;
      if (!femaleVoiceFallbackReported) {
        femaleVoiceFallbackReported = true;
        statusMessage.textContent = voiceChoice.hasEnglishVoices
          ? "No clearly identified female English voice is installed; using your device's default voice."
          : "No female English voice is available yet; using your device's default voice.";
      }
    }
    utterance.rate = 0.95;
    if (voiceChoice.voice) utterance.pitch = 1.05;
    activeUtterance = utterance;

    utterance.onstart = () => {
      if (activeUtterance === utterance) setCompanionState("speaking", null);
    };
    utterance.onend = () => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      setCompanionState("idle");
    };
    utterance.onerror = (event) => {
      if (activeUtterance !== utterance) return;
      activeUtterance = null;
      setCompanionState("idle");
      if (event.error !== "canceled" && event.error !== "interrupted") {
        statusMessage.textContent =
          "Voice playback failed. You can still read Sneha's message.";
      }
    };

    window.speechSynthesis.speak(utterance);
    statusMessage.textContent = status;
  } catch {
    activeUtterance = null;
    setCompanionState("idle");
    statusMessage.textContent = `${status} Browser voice playback failed.`;
  }
}

async function speakAssistantMessage(content) {
  if (!voiceEnabled || !voicePlaybackSupported || !content) return;

  stopVoicePlayback();
  const requestId = voiceRequestId;
  setCompanionState("thinking");
  try {
    const response = await fetch("/api/voice/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: content }),
    });
    if (requestId !== voiceRequestId || !voiceEnabled) return;
    if (!response.ok) {
      const data = await response.json();
      const detail =
        typeof data.detail === "string" ? data.detail : "Azure Speech is unavailable.";
      speakWithBrowser(
        content,
        `Using your browser voice because Azure Speech is unavailable. ${detail}`,
      );
      return;
    }

    const audioBlob = await response.blob();
    if (requestId !== voiceRequestId || !voiceEnabled) return;
    activeAudioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(activeAudioUrl);
    activeAudio = audio;
    audio.addEventListener("play", () => {
      if (activeAudio === audio) setCompanionState("speaking", null);
    });
    audio.addEventListener("ended", () => {
      if (activeAudio !== audio) return;
      activeAudio = null;
      if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
      activeAudioUrl = null;
      setCompanionState("idle");
    }, { once: true });
    audio.addEventListener("error", () => {
      if (activeAudio !== audio) return;
      activeAudio = null;
      if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
      activeAudioUrl = null;
      speakWithBrowser(
        content,
        "Azure audio could not be played. Using your browser voice instead.",
      );
    }, { once: true });
    await audio.play();
  } catch {
    if (requestId !== voiceRequestId || !voiceEnabled) return;
    if (activeAudio) activeAudio.pause();
    activeAudio = null;
    if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
    activeAudioUrl = null;
    speakWithBrowser(
      content,
      "Could not reach Azure Speech. Using your browser voice instead.",
    );
  }
}

function updateVoiceControl() {
  voiceToggle.disabled = !voicePlaybackSupported;
  voiceToggle.setAttribute("aria-pressed", String(voiceEnabled));
  if (!voicePlaybackSupported) {
    voiceToggle.textContent = "Voice unavailable";
    voiceToggle.title = "This browser does not support spoken replies.";
    return;
  }

  voiceToggle.textContent = voiceEnabled ? "Voice: On" : "Voice: Off";
  voiceToggle.setAttribute(
    "aria-label",
    voiceEnabled ? "Turn voice off" : "Turn voice on",
  );
}

function toggleVoice() {
  if (!voicePlaybackSupported) return;

  voiceEnabled = !voiceEnabled;
  updateVoiceControl();
  if (!voiceEnabled) {
    stopVoicePlayback();
    statusMessage.textContent = "Voice replies are off.";
    return;
  }

  statusMessage.textContent = "Voice replies are on.";
  const assistantMessages = messages.querySelectorAll(".message-assistant");
  const latestMessage = assistantMessages[assistantMessages.length - 1];
  if (latestMessage) speakAssistantMessage(latestMessage.textContent);
}

function addMessage(role, content) {
  if (!content) return;
  if (role === "assistant") {
    setCompanionState("speaking");
  }
  const message = document.createElement("p");
  message.className = `message message-${role}`;
  message.textContent = content;
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  if (role === "assistant") speakAssistantMessage(content);
}

function addAnswerOptions(options) {
  if (!Array.isArray(options) || options.length === 0) return;

  const group = document.createElement("div");
  group.className = "answer-options";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Choose an answer");

  for (const option of options) {
    const button = document.createElement("button");
    button.className = "answer-option";
    button.type = "button";
    button.textContent = option;
    button.addEventListener("click", () => sendMessage(option));
    group.append(button);
  }

  messages.append(group);
  messages.scrollTop = messages.scrollHeight;
}

function disableAnswerOptions(disabled) {
  document.querySelectorAll(".answer-option").forEach((button) => {
    button.disabled = disabled;
  });
}

function updateProgress(data) {
  const progress = data.progress;
  if (!progress) return;
  progressBar.value = progress.percentage;
  progressLabel.textContent =
    `${progress.collected} of ${progress.total} questions`;
  stageLabel.textContent = data.stage || "In progress";
}

function showScreeningResult(screeningResult, panel = resultPanel) {
  if (!screeningResult) return;

  panel.replaceChildren();
  panel.hidden = false;

  const heading = document.createElement("h3");
  heading.textContent = screeningResult.status === "safety_interrupted"
    ? "Screening stopped for safety"
    : screeningResult.status === "success"
      ? "Screening result — not a diagnosis"
      : "Screening result unavailable";
  panel.append(heading);

  if (screeningResult.status !== "success") {
    const failure = document.createElement("p");
    failure.textContent =
      screeningResult.message || "The screening result is unavailable.";
    panel.append(failure);
    return;
  }

  const prediction = screeningResult.prediction;
  const model = screeningResult.model_info?.model;
  if (model) {
    const modelLabel = document.createElement("p");
    modelLabel.textContent = `Model: ${model}`;
    panel.append(modelLabel);
  }

  const category = document.createElement("p");
  category.textContent = `Screening category: ${prediction.label}`;
  panel.append(category);

  const depressedProbability =
    screeningResult.class_probabilities?.Depressed ??
    prediction.probability;
  if (typeof depressedProbability === "number") {
    const probability = document.createElement("p");
    probability.textContent =
      `Model-estimated probability for the Depressed screening class: ` +
      `${(depressedProbability * 100).toFixed(1)}%.`;
    panel.append(probability);
  }

  const notDepressedProbability =
    screeningResult.class_probabilities?.["Not Depressed"];
  if (typeof notDepressedProbability === "number") {
    const probability = document.createElement("p");
    probability.textContent =
      `Model-estimated probability for the Not Depressed screening class: ` +
      `${(notDepressedProbability * 100).toFixed(1)}%.`;
    panel.append(probability);
  }

  if (prediction.imputed_features?.length) {
    const imputed = document.createElement("p");
    imputed.textContent =
      "The saved preprocessing pipeline handled unanswered fields: " +
      prediction.imputed_features.join(", ") +
      ". These are imputed values, not answers you provided.";
    panel.append(imputed);
  }

  const disclaimer = document.createElement("p");
  disclaimer.textContent =
    "This is a model-based screening estimate, not a clinical diagnosis. " +
    "Please discuss concerns with a qualified healthcare professional.";
  panel.append(disclaimer);

  const evaluationCaveat = document.createElement("p");
  evaluationCaveat.className = "disclaimer";
  evaluationCaveat.textContent =
    "Important limitation: in the model's dataset, the target label matches " +
    "a cutoff on the same nine symptom scores used as inputs. Its high test " +
    "performance therefore does not establish independent clinical prediction.";
  panel.append(evaluationCaveat);
}

function renderScreeningForm(fields) {
  const groupLabels = {
    numeric: "Background information",
    categorical: "Home and social context",
    symptom: "Wellbeing and symptoms",
  };
  const groups = new Map();

  for (const field of fields) {
    if (!groups.has(field.group)) {
      const section = document.createElement("fieldset");
      section.className = "form-section";
      const legend = document.createElement("legend");
      legend.textContent = groupLabels[field.group] || field.group;
      section.append(legend);
      groups.set(field.group, section);
      formFields.append(section);
    }

    const wrapper = document.createElement("div");
    wrapper.className = "form-field";
    const label = document.createElement("label");
    const inputId = `field-${formFieldDefinitions.length}`;
    label.htmlFor = inputId;
    label.textContent = field.label;

    let input;
    if (field.type === "category" || field.type === "ordinal") {
      input = document.createElement("select");
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Choose an answer";
      placeholder.disabled = true;
      placeholder.selected = true;
      input.append(placeholder);

      for (const option of field.options) {
        const optionElement = document.createElement("option");
        if (typeof option === "object") {
          optionElement.value = option.value;
          optionElement.textContent = option.label;
        } else {
          optionElement.value = option;
          optionElement.textContent = option;
        }
        input.append(optionElement);
      }
    } else {
      input = document.createElement("input");
      input.type = "number";
      if (field.min !== undefined) input.min = field.min;
      if (field.max !== undefined) input.max = field.max;
      if (field.step !== undefined) input.step = field.step;
    }

    input.id = inputId;
    input.name = field.name;
    input.required = true;
    wrapper.append(label, input);
    groups.get(field.group).append(wrapper);
    formFieldDefinitions.push(field);
  }

  const loadingMessage = document.querySelector("#form-loading");
  if (loadingMessage) loadingMessage.remove();
}

async function loadScreeningForm() {
  if (formSchemaLoaded) return;

  const response = await fetch("/api/form/schema");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail),
    );
  }
  if (!Array.isArray(data.fields) || data.fields.length === 0) {
    throw new Error("The screening questions could not be loaded.");
  }

  renderScreeningForm(data.fields);
  formSchemaLoaded = true;
}

async function openScreeningForm() {
  startFormButton.disabled = true;
  formStatusMessage.textContent = "Loading questions...";
  introCard.hidden = true;
  formPanel.hidden = false;
  try {
    await loadScreeningForm();
    formStatusMessage.textContent = "";
  } catch (error) {
    formStatusMessage.textContent = error.message;
  } finally {
    startFormButton.disabled = false;
  }
}

async function submitScreeningForm(event) {
  event.preventDefault();
  if (!screeningForm.reportValidity()) return;

  const answers = {};
  for (const field of formFieldDefinitions) {
    const value = screeningForm.elements.namedItem(field.name).value;
    answers[field.name] =
      field.type === "number" || field.type === "ordinal"
        ? Number(value)
        : value;
  }

  formSubmitButton.disabled = true;
  formStatusMessage.textContent = "Calculating your screening estimate...";
  formResultPanel.hidden = true;
  try {
    const data = await postJson("/api/form/predict", { answers });
    formStatusMessage.textContent = "";
    showScreeningResult(data.screening_result, formResultPanel);
  } catch (error) {
    formStatusMessage.textContent = error.message;
  } finally {
    formSubmitButton.disabled = false;
  }
}

function returnToStart() {
  formPanel.hidden = true;
  conversationPanel.hidden = true;
  introCard.hidden = false;
  formStatusMessage.textContent = "";
}

function updateControls(data) {
  currentResponse = data;
  const awaitingSafetyAnswer = data.safety_follow_up === true;
  const ended = data.completed || (data.screening_stopped && !awaitingSafetyAnswer);
  replyInput.disabled = requestPending || ended;
  sendButton.disabled = requestPending || ended;
  if (data.screening_stopped && awaitingSafetyAnswer) {
    statusMessage.textContent =
      "Screening is stopped. Please answer Sneha's immediate-safety question.";
  } else if (data.screening_stopped) {
    statusMessage.textContent = "This screening session has ended.";
  } else if (data.completed) {
    statusMessage.textContent = "The interview is complete.";
  }
}

function renderResponse(data) {
  disableAnswerOptions(true);
  document.querySelectorAll(".answer-options").forEach((group) => group.remove());
  document.querySelectorAll(".answer-hint").forEach((hint) => hint.remove());
  updateProgress(data);
  addMessage("assistant", data.assistant_message);
  addAnswerOptions(data.answer_options);
  showScreeningResult(data.screening_result);
  updateControls(data);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail),
    );
  }
  return data;
}

async function startConversation() {
  requestPending = true;
  startButton.disabled = true;
  statusMessage.textContent = "Starting your conversation...";
  setCompanionState("thinking");
  try {
    const data = await postJson("/api/conversation/start", {});
    sessionId = data.session_id;
    introCard.hidden = true;
    conversationPanel.hidden = false;
    renderResponse(data);
  } catch (error) {
    statusMessage.textContent = error.message;
    startButton.disabled = false;
    setCompanionState("idle");
  } finally {
    requestPending = false;
    if (!conversationPanel.hidden) {
      updateControls(currentResponse);
      if (!replyInput.disabled) replyInput.focus();
    }
  }
}

async function sendMessage(message) {
  if (!sessionId || !message || requestPending) return;

  stopVoicePlayback();
  addMessage("user", message);
  replyInput.value = "";
  disableAnswerOptions(true);
  requestPending = true;
  replyInput.disabled = true;
  sendButton.disabled = true;
  statusMessage.textContent = "Sneha is preparing a response...";
  setCompanionState("thinking");

  try {
    const data = await postJson("/api/conversation/respond", {
      session_id: sessionId,
      message,
    });
    renderResponse(data);
  } catch (error) {
    statusMessage.textContent = error.message;
    replyInput.value = message;
    setCompanionState("idle");
  } finally {
    requestPending = false;
    updateControls(currentResponse);
    if (!replyInput.disabled) replyInput.focus();
  }
}

async function sendReply(event) {
  event.preventDefault();
  await sendMessage(replyInput.value.trim());
}

startButton.addEventListener("click", startConversation);
voiceToggle.addEventListener("click", toggleVoice);
startFormButton.addEventListener("click", openScreeningForm);
screeningForm.addEventListener("submit", submitScreeningForm);
backToIntroButton.addEventListener("click", returnToStart);
replyForm.addEventListener("submit", sendReply);

updateVoiceControl();
