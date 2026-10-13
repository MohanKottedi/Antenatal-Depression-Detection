"use strict";

const startButton = document.querySelector("#start-button");
const introCard = document.querySelector(".intro-card");
const conversationPanel = document.querySelector("#conversation-panel");
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

function addMessage(role, content) {
  if (!content) return;
  const message = document.createElement("p");
  message.className = `message message-${role}`;
  message.textContent = content;
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
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

function showScreeningResult(screeningResult) {
  if (!screeningResult) return;

  resultPanel.replaceChildren();
  resultPanel.hidden = false;

  const heading = document.createElement("h3");
  heading.textContent = "Screening result — not a diagnosis";
  resultPanel.append(heading);

  if (screeningResult.status !== "success") {
    const failure = document.createElement("p");
    failure.textContent =
      screeningResult.message || "The screening result is unavailable.";
    resultPanel.append(failure);
    return;
  }

  const prediction = screeningResult.prediction;
  const model = screeningResult.model_info?.model;
  if (model) {
    const modelLabel = document.createElement("p");
    modelLabel.textContent = `Model: ${model}`;
    resultPanel.append(modelLabel);
  }

  const category = document.createElement("p");
  category.textContent = `Screening category: ${prediction.label}`;
  resultPanel.append(category);

  const depressedProbability =
    screeningResult.class_probabilities?.Depressed ??
    prediction.probability;
  if (typeof depressedProbability === "number") {
    const probability = document.createElement("p");
    probability.textContent =
      `Model-estimated probability for the Depressed screening class: ` +
      `${(depressedProbability * 100).toFixed(1)}%.`;
    resultPanel.append(probability);
  }

  const notDepressedProbability =
    screeningResult.class_probabilities?.["Not Depressed"];
  if (typeof notDepressedProbability === "number") {
    const probability = document.createElement("p");
    probability.textContent =
      `Model-estimated probability for the Not Depressed screening class: ` +
      `${(notDepressedProbability * 100).toFixed(1)}%.`;
    resultPanel.append(probability);
  }

  if (prediction.imputed_features?.length) {
    const imputed = document.createElement("p");
    imputed.textContent =
      "The saved preprocessing pipeline handled unanswered fields: " +
      prediction.imputed_features.join(", ") +
      ". These are imputed values, not answers you provided.";
    resultPanel.append(imputed);
  }

  const disclaimer = document.createElement("p");
  disclaimer.textContent =
    "This is a model-based screening estimate, not a clinical diagnosis. " +
    "Please discuss concerns with a qualified healthcare professional.";
  resultPanel.append(disclaimer);

  const evaluationCaveat = document.createElement("p");
  evaluationCaveat.className = "disclaimer";
  evaluationCaveat.textContent =
    "Important limitation: in the model's dataset, the target label matches " +
    "a cutoff on the same nine symptom scores used as inputs. Its high test " +
    "performance therefore does not establish independent clinical prediction.";
  resultPanel.append(evaluationCaveat);
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
  try {
    const data = await postJson("/api/conversation/start", {});
    sessionId = data.session_id;
    introCard.hidden = true;
    conversationPanel.hidden = false;
    renderResponse(data);
  } catch (error) {
    statusMessage.textContent = error.message;
    startButton.disabled = false;
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

  addMessage("user", message);
  replyInput.value = "";
  disableAnswerOptions(true);
  requestPending = true;
  replyInput.disabled = true;
  sendButton.disabled = true;
  statusMessage.textContent = "Sneha is preparing a response...";

  try {
    const data = await postJson("/api/conversation/respond", {
      session_id: sessionId,
      message,
    });
    renderResponse(data);
  } catch (error) {
    statusMessage.textContent = error.message;
    replyInput.value = message;
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
replyForm.addEventListener("submit", sendReply);
