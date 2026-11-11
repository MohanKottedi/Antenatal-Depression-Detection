(() => {
  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
  const detectedTimezone = () => {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
    } catch (error) {
      console.error('Could not detect the browser timezone.', error);
      return 'UTC';
    }
  };

  async function api(url, options = {}) {
    const response = await fetch(url, {
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        window.location.href = '/auth/login';
      }
      throw new Error(data.detail || 'Could not complete the request.');
    }
    return data;
  }

  const viewNames = new Set([
    'overview',
    'pregnancy',
    'mood',
    'mental-health',
    'assistant',
    'relaxation',
    'nutrition',
    'emergency',
  ]);

  function setActiveView(viewName, updateHistory = false) {
    const activeView = viewNames.has(viewName) ? viewName : 'overview';
    document.querySelectorAll('[data-view-page]').forEach((view) => {
      view.hidden = view.dataset.viewPage !== activeView;
    });
    document.querySelectorAll('[data-view-target]').forEach((link) => {
      const selected = link.dataset.viewTarget === activeView;
      link.classList.toggle('active', selected);
      if (selected) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });
    if (updateHistory) {
      window.history.pushState(null, '', `#${activeView}`);
    }
    document.querySelector('.dashboard-main').scrollTo({ top: 0, behavior: 'smooth' });
  }

  function initializeNavigation() {
    document.querySelectorAll('[data-view-target]').forEach((link) => {
      link.addEventListener('click', (event) => {
        event.preventDefault();
        setActiveView(link.dataset.viewTarget, true);
      });
    });

    window.addEventListener('popstate', () => {
      setActiveView(window.location.hash.slice(1));
    });

    const initialView = window.location.hash.slice(1) || 'overview';
    setActiveView(initialView);
    if (!window.location.hash) {
      window.history.replaceState(null, '', '#overview');
    }
  }

  function showStatus(selector, text, isError = false) {
    const element = $(selector);
    if (!element) return;
    element.textContent = text;
    element.classList.toggle('error', isError);
  }

  function linkList(items) {
    return (items || []).map((item) => {
      const url = String(item.url || '');
      if (!url.startsWith('https://')) return '';
      return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.label)}</a>`;
    }).join('');
  }

  async function loadPregnancy() {
    const data = await api('/api/pregnancy');
    const profile = data.profile || {};
    const tracker = data.tracker || {};
    const form = $('#pregnancy-profile-form');
    for (const field of ['lmp_date', 'confirmed_due_date', 'follow_up_reminder_date', 'dietary_preference', 'food_allergies']) {
      form.elements.namedItem(field).value = profile[field] || (field === 'dietary_preference' ? 'no_preference' : '');
    }
    form.elements.namedItem('reminder_time').value = profile.reminder_time || '15:41';
    form.elements.namedItem('weekly_tracker_reminders').checked = profile.weekly_tracker_reminders !== false;
    form.elements.namedItem('screening_reminders').checked = profile.screening_reminders !== false;
    $('#reminder-timezone').textContent = profile.reminder_timezone || detectedTimezone();

    const trimester = $('#pregnancy-trimester');
    const progress = $('#pregnancy-progress');
    if (tracker.gestational_week !== null && tracker.gestational_week !== undefined) {
      trimester.textContent = `Trimester ${tracker.trimester}`;
      const expectedDueDate = tracker.due_date
        ? new Date(`${tracker.due_date}T12:00:00`).toLocaleDateString(undefined, { dateStyle: 'medium' })
        : 'Not available';
      progress.innerHTML = `
        <div class="pregnancy-stats">
          <strong>${Number(tracker.gestational_week)} weeks <span>${Number(tracker.gestational_day)} days</span></strong>
          <span>Due date ${escapeHtml(expectedDueDate)}${tracker.confirmed_due_date ? ' · clinician-confirmed' : ' · LMP estimate'}</span>
        </div>
        <div class="pregnancy-progress-track" role="progressbar" aria-label="Estimated pregnancy progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Number(tracker.progress_percentage || 0)}">
          <span style="width:${Math.max(0, Math.min(100, Number(tracker.progress_percentage || 0)))}%"></span>
        </div>
        <p>${escapeHtml(tracker.milestone || '')}</p>
        ${(tracker.upcoming_milestones || []).length ? `<strong>Helpful next steps</strong><ul>${tracker.upcoming_milestones.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
        <p class="small-muted">${escapeHtml(tracker.notice || '')}</p>
        <div class="reference-list">${linkList(tracker.references)}</div>
      `;
    } else {
      trimester.textContent = 'Dates not set';
      progress.innerHTML = `
        <p>Enter your last menstrual period or clinician-confirmed due date to see an estimate.</p>
        <p class="small-muted">${escapeHtml(tracker.notice || '')}</p>
        <div class="reference-list">${linkList(tracker.references)}</div>
      `;
    }
  }

  async function savePregnancy(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const payload = {};
    for (const field of ['lmp_date', 'confirmed_due_date', 'follow_up_reminder_date', 'dietary_preference', 'food_allergies']) {
      const value = String(formData.get(field) || '').trim();
      payload[field] = value || null;
    }
    payload.reminder_time = String(formData.get('reminder_time') || '15:41');
    payload.reminder_timezone = detectedTimezone();
    payload.weekly_tracker_reminders = form.elements.namedItem('weekly_tracker_reminders').checked;
    payload.screening_reminders = form.elements.namedItem('screening_reminders').checked;
    try {
      await api('/api/pregnancy', { method: 'PUT', body: JSON.stringify(payload) });
      showStatus('#pregnancy-status', 'Pregnancy dates, preferences, and local reminder time saved.');
      await Promise.all([loadPregnancy(), loadMentalHealth(), loadNutrition()]);
    } catch (error) {
      showStatus('#pregnancy-status', error.message, true);
    }

    function decodeApplicationServerKey(value) {
      const padding = '='.repeat((4 - (value.length % 4)) % 4);
      const base64 = (value + padding).replace(/-/g, '+').replace(/_/g, '/');
      return Uint8Array.from(window.atob(base64), (character) => character.charCodeAt(0));
    }

    async function loadNotificationStatus() {
      const status = $('#notification-status');
      const enableButton = $('#enable-notifications');
      const disableButton = $('#disable-notifications');
      const data = await api('/api/notifications/status');
      if (!data.push_configured) {
        status.textContent = 'Background push is not configured by the server yet. Ask the administrator to configure VAPID keys and start the notification worker.';
        enableButton.disabled = true;
        disableButton.hidden = true;
        return;
      }
      const supported = 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;
      if (!supported) {
        status.textContent = 'This browser does not support background push notifications. The selected reminder time is still saved in your account.';
        enableButton.disabled = true;
        disableButton.hidden = true;
        return;
      }
      const registration = await navigator.serviceWorker.getRegistration('/');
      const subscription = registration ? await registration.pushManager.getSubscription() : null;
      enableButton.disabled = Notification.permission === 'denied';
      disableButton.hidden = !subscription;
      if (Notification.permission === 'denied') {
        status.textContent = 'Notifications are blocked by this browser. Allow them in the site settings, then try again.';
      } else if (subscription && data.subscribed) {
        status.textContent = 'Background reminders are enabled on this device. Weekly tracker updates and saved screening reminders use your selected local time.';
      } else {
        status.textContent = 'Notifications are off on this device. Enable them to receive weekly tracker and screening reminders while the dashboard is closed.';
      }
    }

    async function enableNotifications() {
      const button = $('#enable-notifications');
      button.disabled = true;
      try {
        const settings = await api('/api/notifications/status');
        if (!settings.push_configured || !settings.public_key) {
          throw new Error('Background push is not configured by the server.');
        }
        if (!('serviceWorker' in navigator) || !('PushManager' in window) || !('Notification' in window)) {
          throw new Error('This browser does not support background push notifications.');
        }
        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
          throw new Error(permission === 'denied'
            ? 'Notifications were blocked. Allow them in this site’s browser settings to continue.'
            : 'Notification permission was not granted.');
        }
        const reminderTime = $('#pregnancy-profile-form').elements.namedItem('reminder_time').value || '15:41';
        const timezone = detectedTimezone();
        await api('/api/pregnancy', {
          method: 'PUT',
          body: JSON.stringify({ reminder_time: reminderTime, reminder_timezone: timezone }),
        });
        const registration = await navigator.serviceWorker.register('/service-worker.js');
        let subscription = await registration.pushManager.getSubscription();
        if (!subscription) {
          subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: decodeApplicationServerKey(settings.public_key),
          });
        }
        await api('/api/notifications/subscription', {
          method: 'POST',
          body: JSON.stringify({
            ...subscription.toJSON(),
            reminder_time: reminderTime,
            reminder_timezone: timezone,
          }),
        });
        $('#reminder-timezone').textContent = timezone;
        showStatus('#notification-status', 'Background reminders are enabled on this device.');
        await loadNotificationStatus();
      } catch (error) {
        showStatus('#notification-status', error.message, true);
        button.disabled = false;
      }
    }

    async function disableNotifications() {
      const button = $('#disable-notifications');
      button.disabled = true;
      try {
        const registration = await navigator.serviceWorker.getRegistration('/');
        const subscription = registration ? await registration.pushManager.getSubscription() : null;
        if (subscription) {
          await api('/api/notifications/subscription', {
            method: 'DELETE',
            body: JSON.stringify({ endpoint: subscription.endpoint }),
          });
          await subscription.unsubscribe();
        }
        showStatus('#notification-status', 'Background reminders are turned off on this device.');
        await loadNotificationStatus();
      } catch (error) {
        showStatus('#notification-status', error.message, true);
        button.disabled = false;
      }
    }
  }

  function renderMoodHistory(entries) {
    const list = $('#mood-history');
    if (!entries.length) {
      list.innerHTML = '<p class="empty-state">No mood entries yet. Your check-ins are visible only in your account.</p>';
      return;
    }
    list.innerHTML = entries.slice(0, 14).map((entry) => `
      <article class="mood-history-row">
        <div>
          <strong>${escapeHtml(entry.mood[0].toUpperCase() + entry.mood.slice(1))} · ${escapeHtml(new Date(`${entry.entry_date}T12:00:00`).toLocaleDateString())}</strong>
          <span>Stress ${entry.stress_level ?? 'not recorded'} · Sleep ${entry.sleep_hours ?? 'not recorded'} hours</span>
          ${entry.journal ? `<p>${escapeHtml(entry.journal)}</p>` : ''}
        </div>
        <button class="secondary-button danger-button" type="button" data-delete-mood="${Number(entry.id)}" aria-label="Delete mood entry">Delete</button>
      </article>
    `).join('');
    list.querySelectorAll('[data-delete-mood]').forEach((button) => {
      button.addEventListener('click', async () => {
        if (!window.confirm('Delete this mood entry?')) return;
        try {
          await api(`/api/mood/entries/${button.dataset.deleteMood}`, { method: 'DELETE' });
          await Promise.all([loadMood(), loadMentalHealth()]);
        } catch (error) {
          showStatus('#mood-status', error.message, true);
        }
      });
    });
  }

  function renderMoodChart(entries) {
    const chart = $('#mood-chart');
    const recent = entries
      .filter((entry) => entry.entry_date >= new Date(Date.now() - 6 * 86400000).toISOString().slice(0, 10))
      .slice()
      .reverse();
    if (!recent.length) {
      chart.innerHTML = '<p class="empty-state">Your check-ins will appear here over the last seven days.</p>';
      return;
    }
    chart.innerHTML = recent.map((entry) => `
      <div class="trend-row">
        <span>${escapeHtml(new Date(`${entry.entry_date}T12:00:00`).toLocaleDateString(undefined, { weekday: 'short' }))}</span>
        <strong>${escapeHtml(entry.mood)}</strong>
        <div class="trend-track" aria-label="Stress level ${entry.stress_level ?? 'not recorded'} out of 5">
          <span style="width:${Math.max(0, Math.min(100, Number(entry.stress_level || 0) * 20))}%"></span>
        </div>
        <small>${entry.stress_level ? `Stress ${entry.stress_level}/5` : 'Stress n/a'}${entry.sleep_hours !== null && entry.sleep_hours !== undefined ? ` · ${entry.sleep_hours}h sleep` : ''}</small>
      </div>
    `).join('');
  }

  let calendarMonth = new Date();
  function renderMoodCalendar(entries) {
    const year = calendarMonth.getFullYear();
    const month = calendarMonth.getMonth();
    const monthStart = new Date(year, month, 1);
    const firstWeekday = monthStart.getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const byDate = new Map(entries.map((entry) => [entry.entry_date, entry]));
    const cells = [];
    for (let index = 0; index < firstWeekday; index += 1) cells.push('<td aria-hidden="true"></td>');
    for (let day = 1; day <= daysInMonth; day += 1) {
      const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const entry = byDate.get(key);
      cells.push(`<td class="${entry ? 'has-mood' : ''}" ${entry ? `aria-label="${escapeHtml(key)}: ${escapeHtml(entry.mood)}"` : ''}><span>${day}</span>${entry ? `<small>${escapeHtml(entry.mood)}</small>` : ''}</td>`);
    }
    while (cells.length % 7) cells.push('<td aria-hidden="true"></td>');
    const weeks = [];
    for (let index = 0; index < cells.length; index += 7) {
      weeks.push(`<tr>${cells.slice(index, index + 7).join('')}</tr>`);
    }
    $('#mood-calendar-title').textContent = monthStart.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
    $('#mood-calendar').innerHTML = `
      <table class="mood-calendar-table">
        <thead><tr>${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => `<th scope="col">${day}</th>`).join('')}</tr></thead>
        <tbody>${weeks.join('')}</tbody>
      </table>
    `;
  }

  function renderMonthlyMoodChart(entries) {
    const chart = $('#mood-monthly-chart');
    const buckets = new Map();
    for (const entry of entries) {
      const month = entry.entry_date.slice(0, 7);
      const bucket = buckets.get(month) || { count: 0, stressTotal: 0, stressCount: 0 };
      bucket.count += 1;
      if (entry.stress_level !== null && entry.stress_level !== undefined) {
        bucket.stressTotal += Number(entry.stress_level);
        bucket.stressCount += 1;
      }
      buckets.set(month, bucket);
    }
    const months = [...buckets.entries()].sort(([left], [right]) => left.localeCompare(right));
    if (!months.length) {
      chart.innerHTML = '<p class="empty-state">Monthly trends will appear as you add mood entries.</p>';
      return;
    }
    const max = Math.max(...months.map(([, item]) => item.count), 1);
    chart.innerHTML = months.map(([month, item]) => `
      <div class="trend-row monthly-mood-row">
        <span>${escapeHtml(new Date(`${month}-15T12:00:00`).toLocaleDateString(undefined, { month: 'short' }))}</span>
        <div class="trend-track"><span style="width:${(item.count / max) * 100}%"></span></div>
        <strong>${item.count} check-in${item.count === 1 ? '' : 's'}</strong>
        <small>${item.stressCount ? `Average stress ${ (item.stressTotal / item.stressCount).toFixed(1) } / 5` : 'Stress not recorded'}</small>
      </div>
    `).join('');
  }

  async function loadMood() {
    const data = await api('/api/mood/entries?days=365');
    renderMoodHistory(data.entries || []);
    renderMoodChart(data.entries || []);
    renderMoodCalendar(data.entries || []);
    renderMonthlyMoodChart(data.entries || []);
  }

  async function saveMood(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const stress = formData.get('stress_level');
    const sleep = formData.get('sleep_hours');
    const payload = {
      mood: formData.get('mood'),
      stress_level: stress === '' ? null : Number(stress),
      sleep_hours: sleep === '' ? null : Number(sleep),
      journal: String(formData.get('journal') || '').trim() || null,
    };
    try {
      await api('/api/mood/entries', { method: 'POST', body: JSON.stringify(payload) });
      showStatus('#mood-status', 'Today’s mood check-in is saved. You can update it again today.');
      form.elements.namedItem('journal').value = '';
      await Promise.all([loadMood(), loadMentalHealth()]);
    } catch (error) {
      showStatus('#mood-status', error.message, true);
    }
  }

  function renderMonthlyScreenings(screenings) {
    const chart = $('#screening-chart');
    const buckets = new Map();
    screenings.filter((item) => item.status === 'completed').forEach((item) => {
      const key = item.date.slice(0, 7);
      buckets.set(key, (buckets.get(key) || 0) + 1);
    });
    const rows = [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b)).slice(-6);
    if (!rows.length) {
      chart.innerHTML = '<p class="empty-state">Completed assessments will be charted here.</p>';
      return;
    }
    const max = Math.max(...rows.map(([, count]) => count), 1);
    chart.innerHTML = rows.map(([month, count]) => `
      <div class="trend-row">
        <span>${escapeHtml(new Date(`${month}-15T12:00:00`).toLocaleDateString(undefined, { month: 'short', year: 'numeric' }))}</span>
        <div class="trend-track"><span style="width:${(count / max) * 100}%"></span></div>
        <strong>${count} assessment${count === 1 ? '' : 's'}</strong>
      </div>
    `).join('');
  }

  function renderAssessmentTrends(screenings) {
    const recent = screenings.slice().reverse().slice(-12);
    const renderBars = (selector, getValue, maxValue, suffix, emptyText) => {
      const chart = $(selector);
      const available = recent.map((entry) => ({ entry, value: getValue(entry) }))
        .filter((item) => item.value !== null && item.value !== undefined && Number.isFinite(Number(item.value)));
      if (!available.length) {
        chart.innerHTML = `<p class="empty-state">${escapeHtml(emptyText)}</p>`;
        return;
      }
      chart.innerHTML = `<div class="assessment-bars" role="list">${available.map(({ entry, value }) => {
        const numeric = Math.max(0, Math.min(maxValue, Number(value)));
        const dateLabel = new Date(entry.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        const amount = `${suffix === '%' ? (numeric * 100).toFixed(0) : numeric.toFixed(1)}${suffix}`;
        return `
          <button class="assessment-bar-item" type="button" role="listitem" title="${escapeHtml(dateLabel)} · ${escapeHtml(amount)}" aria-label="${escapeHtml(dateLabel)}: ${escapeHtml(amount)}" data-open-screening="${Number(entry.id)}">
            <span class="assessment-bar-value">${escapeHtml(amount)}</span>
            <span class="assessment-bar-track"><span style="height:${(numeric / maxValue) * 100}%"></span></span>
            <span class="assessment-bar-date">${escapeHtml(dateLabel)}</span>
          </button>
        `;
      }).join('')}</div>`;
    };
    renderBars(
      '#screening-score-chart',
      (entry) => entry.symptom_item_total,
      27,
      '',
      'Symptom response sums will appear after a complete questionnaire is saved.',
    );
    renderBars(
      '#screening-model-chart',
      (entry) => entry.model_probability,
      1,
      '%',
      'Model probabilities will appear when a prediction is available.',
    );
    document.querySelectorAll('#screening-score-chart [data-open-screening], #screening-model-chart [data-open-screening]').forEach((button) => {
      button.addEventListener('click', () => {
        setActiveView('overview', true);
        document.querySelector(`#screening-list [data-session-id="${button.dataset.openScreening}"]`)?.click();
        document.querySelector('#screening-list')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
  }

  async function loadMentalHealth() {
    const data = await api('/api/mental-health/progress');
    const reminder = data.follow_up_reminder_date;
    const reminderNotice = reminder
      ? `<p class="info-box">${new Date(`${reminder}T12:00:00`) < new Date(new Date().toDateString())
        ? 'Your follow-up screening reminder is due.'
        : 'Your follow-up screening reminder is set for'} ${escapeHtml(new Date(`${reminder}T12:00:00`).toLocaleDateString())}. If push is enabled in Pregnancy tracker, Sneha will send a notification at your selected local time.</p>`
      : '<p class="info-box">Set an optional follow-up date in Pregnancy tracker. If push is enabled, Sneha can notify you at your selected local time.</p>';
    $('#mental-health-notices').innerHTML = `${reminderNotice}${(data.notices || []).map((notice) => `<p class="small-muted">${escapeHtml(notice)}</p>`).join('')}`;
    const screenings = data.screenings || [];
    $('#screening-progress').innerHTML = screenings.length ? screenings.map((entry) => {
      const dateLabel = new Date(entry.date).toLocaleDateString(undefined, { dateStyle: 'medium' });
      const score = entry.symptom_item_total === null || entry.symptom_item_total === undefined
        ? 'Not available'
        : `${entry.symptom_item_total} / 27 symptom-item sum`;
      const result = entry.model_prediction
        ? `Model estimate: ${entry.model_prediction}${entry.model_probability !== null && entry.model_probability !== undefined ? ` · ${(Number(entry.model_probability) * 100).toFixed(1)}% estimated probability` : ''}`
        : 'No model prediction saved';
      return `
        <article class="progress-entry">
          <div class="detail-header"><strong>Assessment · ${escapeHtml(dateLabel)}</strong><span>${escapeHtml(entry.status)}</span></div>
          <p>Questionnaire response sum: ${escapeHtml(score)}</p>
          <p>${escapeHtml(result)}</p>
          <button class="secondary-button" type="button" data-open-screening="${Number(entry.id)}">View saved answers</button>
        </article>
      `;
    }).join('') : '<p class="empty-state">Your completed assessments and results will appear here.</p>';
    $('#screening-progress').querySelectorAll('[data-open-screening]').forEach((button) => {
      button.addEventListener('click', () => {
        setActiveView('overview', true);
        document.querySelector(`#screening-list [data-session-id="${button.dataset.openScreening}"]`)?.click();
        document.querySelector('#screening-list')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });
    renderMonthlyScreenings(screenings);
    renderAssessmentTrends(screenings);
  }

  function renderAssistantMessages(messages) {
    const container = $('#assistant-messages');
    container.innerHTML = '';
    if (!messages.length) {
      container.innerHTML = '<p class="empty-state">Ask a general pregnancy or wellbeing question to begin.</p>';
      return;
    }
    for (const message of messages) {
      const article = document.createElement('article');
      article.className = `assistant-message assistant-${message.role}`;
      const label = document.createElement('strong');
      label.textContent = message.role === 'user' ? 'You' : 'Sneha · general information';
      const content = document.createElement('p');
      content.textContent = message.content;
      article.append(label, content);
      container.append(article);
    }
    container.scrollTop = container.scrollHeight;
  }

  let activeConversationId = null;
  let hydrationReminderTimer = null;
  async function loadConversationList(selectId = null) {
    const data = await api('/api/assistant/conversations');
    const select = $('#assistant-conversation-select');
    select.innerHTML = '';
    for (const conversation of data.conversations || []) {
      const option = document.createElement('option');
      option.value = String(conversation.id);
      option.textContent = `${conversation.title} · ${new Date(conversation.updated_at).toLocaleDateString()}`;
      select.append(option);
    }
    activeConversationId = selectId || (data.conversations || [])[0]?.id || null;
    if (!activeConversationId) {
      const created = await api('/api/assistant/conversations', { method: 'POST' });
      activeConversationId = created.conversation.id;
      return loadConversationList(activeConversationId);
    }
    select.value = String(activeConversationId);
    await loadConversation();
  }

  async function loadConversation() {
    if (!activeConversationId) return;
    const data = await api(`/api/assistant/conversations/${activeConversationId}`);
    renderAssistantMessages(data.messages || []);
  }

  async function sendAssistantMessage(event) {
    event.preventDefault();
    const input = $('#assistant-input');
    const message = input.value.trim();
    if (!message || !activeConversationId) return;
    $('#assistant-send').disabled = true;
    showStatus('#assistant-status', 'Sneha is thinking…');
    try {
      await api(`/api/assistant/conversations/${activeConversationId}/messages`, {
        method: 'POST',
        body: JSON.stringify({ message }),
      });
      input.value = '';
      showStatus('#assistant-status', '');
      await Promise.all([loadConversationList(activeConversationId), loadConversation()]);
    } catch (error) {
      showStatus('#assistant-status', error.message, true);
      await loadConversation();
    } finally {
      $('#assistant-send').disabled = false;
    }
  }

  function startMeditation(activity, card) {
    const selector = card.querySelector('select');
    const duration = Number(selector.value);
    const status = card.querySelector('[data-timer-status]');
    const start = card.querySelector('[data-start]');
    const stop = card.querySelector('[data-stop]');
    const breath = card.querySelector('.breathing-orb');
    let remaining = duration;
    let timerId = null;
    let startedAt = null;
    start.disabled = true;
    stop.disabled = false;
    breath.classList.add('breathing-active');
    startedAt = Date.now();
    const update = () => {
      remaining = Math.max(0, duration - Math.floor((Date.now() - startedAt) / 1000));
      status.textContent = `Take your time · ${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')} remaining`;
      if (remaining === 0) {
        window.clearInterval(timerId);
        breath.classList.remove('breathing-active');
        stop.disabled = true;
        start.disabled = false;
        status.textContent = 'Session complete. Saving…';
        api('/api/meditation/sessions', {
          method: 'POST',
          body: JSON.stringify({ activity_id: activity.id, duration_seconds: duration }),
        }).then(() => {
          status.textContent = 'Session complete and recorded. Take a moment to notice how you feel.';
        }).catch((error) => {
          status.textContent = `Session complete, but it could not be recorded: ${error.message}`;
        });
      }
    };
    stop.onclick = () => {
      window.clearInterval(timerId);
      breath.classList.remove('breathing-active');
      status.textContent = 'Session stopped; it was not recorded as complete.';
      start.disabled = false;
      stop.disabled = true;
      stop.onclick = null;
    };
    update();
    timerId = window.setInterval(update, 500);
  }

  async function loadMeditation() {
    const data = await api('/api/meditation/activities');
    $('#meditation-activities').innerHTML = (data.activities || []).map((activity) => `
      <article class="wellbeing-card meditation-card">
        <div class="breathing-orb" aria-hidden="true"></div>
        <h4>${escapeHtml(activity.title)}</h4>
        <p>${escapeHtml(activity.description)}</p>
        <label class="auth-field">Duration
          <select aria-label="${escapeHtml(activity.title)} duration">
            ${activity.durations.map((seconds) => `<option value="${Number(seconds)}">${Math.floor(seconds / 60)} minutes</option>`).join('')}
          </select>
        </label>
        <div class="inline-actions">
          <button class="primary-button" type="button" data-start>Start</button>
          <button class="secondary-button" type="button" data-stop disabled>Stop</button>
        </div>
        <p data-timer-status class="small-muted">${(data.recent_sessions || []).filter((item) => item.activity_id === activity.id).length} sessions completed recently.</p>
      </article>
    `).join('');
    $('#meditation-activities').querySelectorAll('.meditation-card').forEach((card, index) => {
      const activity = data.activities[index];
      card.querySelector('[data-start]').addEventListener('click', () => startMeditation(activity, card));
    });
    showStatus('#meditation-status', data.notice || '');
  }

  async function loadNutrition() {
    const data = await api('/api/nutrition');
    const mealIdeas = data.meal_ideas || {};
    $('#nutrition-content').innerHTML = `
      ${data.trimester ? `<p class="wellbeing-badge">General ideas · trimester ${Number(data.trimester)}</p>` : ''}
      <div class="meal-ideas-grid">
        ${Object.entries(mealIdeas).map(([meal, idea]) => `<div class="info-box"><strong>${escapeHtml(meal[0].toUpperCase() + meal.slice(1))}</strong><p>${escapeHtml(idea)}</p></div>`).join('')}
      </div>
      <p>${escapeHtml(data.guidance || '')}</p>
      <p>${escapeHtml(data.notice || '')}</p>
      <p class="small-muted">Hydration: drink regularly and follow advice from your maternity clinician; fluid needs can vary.</p>
      <div class="reference-list">${linkList(data.references)}</div>
    `;
    $('#ai-meal-ideas').hidden = true;
  }

  async function generateMealIdeas() {
    const checkbox = $('#share-allergies-local');
    if (!checkbox.checked && window.confirm('Your saved allergy notes will not be shared. Continue with general meal inspiration?') === false) return;
    $('#generate-meal-ideas').disabled = true;
    showStatus('#nutrition-status', 'Preparing general meal inspiration…');
    try {
      const data = await api('/api/nutrition/ideas', {
        method: 'POST',
        body: JSON.stringify({
          allow_local_allergy_context: checkbox.checked,
        }),
      });
      const content = $('#ai-meal-ideas');
      content.textContent = data.suggestion;
      content.hidden = false;
      showStatus('#nutrition-status', data.notice || '');
    } catch (error) {
      showStatus('#nutrition-status', error.message, true);
    } finally {
      $('#generate-meal-ideas').disabled = false;
    }
  }

  async function loadEmergency() {
    const data = await api('/api/emergency/resources');
    $('#emergency-guidance').textContent = data.region
      ? `${data.region}: ${data.guidance}`
      : data.guidance;
    const contacts = $('#emergency-contacts');
    contacts.innerHTML = data.contacts.length
      ? data.contacts.map((item) => {
        const contact = String(item.contact || '');
        const safePhone = /^[+()\d\s.-]+$/.test(contact);
        const safeUrl = contact.startsWith('https://');
        const display = safePhone
          ? `<a href="tel:${escapeHtml(contact.replace(/\s/g, ''))}">${escapeHtml(contact)}</a>`
          : safeUrl
            ? `<a href="${escapeHtml(contact)}" target="_blank" rel="noopener noreferrer">${escapeHtml(contact)}</a>`
            : escapeHtml(contact);
        return `<div class="info-row"><strong>${escapeHtml(item.label)}</strong><span>${display}</span></div>`;
      }).join('')
      : '<p class="empty-state">No local contacts have been configured. Please use the emergency number or maternity contact for your own location; do not rely on this page to provide a number.</p>';
    $('#emergency-warning-signs').innerHTML = (data.warning_signs || []).map((sign) => `<li>${escapeHtml(sign)}</li>`).join('');
    $('#emergency-map-notice').textContent = data.map_search.notice;
    $('#emergency-references').innerHTML = linkList(data.references);
  }

  function findNearbyHospital() {
    if (!navigator.geolocation) {
      showStatus('#emergency-map-notice', 'Location search is not supported by this browser.');
      return;
    }
    if (!window.confirm('Your location will be shared with Google Maps to search for maternity hospitals. Continue?')) return;
    const mapWindow = window.open('about:blank', '_blank');
    if (mapWindow) mapWindow.opener = null;
    showStatus('#emergency-map-notice', 'Waiting for your location permission…');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const query = encodeURIComponent('maternity hospital');
        const coordinates = `${position.coords.latitude},${position.coords.longitude}`;
        if (mapWindow) {
          mapWindow.location = `https://www.google.com/maps/search/${query}/@${coordinates},13z`;
        } else {
          showStatus('#emergency-map-notice', 'Your browser blocked the map window. Allow pop-ups and try again.');
          return;
        }
        showStatus('#emergency-map-notice', 'Google Maps opened with your location. Contact local emergency services directly for urgent help.');
      },
      () => {
        mapWindow?.close();
        showStatus('#emergency-map-notice', 'Location was unavailable or permission was not granted. Contact local emergency services or your maternity team directly.');
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 },
    );
  }

  function toggleHydrationReminder() {
    const button = $('#hydration-reminder-toggle');
    const status = $('#hydration-reminder-status');
    if (hydrationReminderTimer !== null) {
      window.clearInterval(hydrationReminderTimer);
      hydrationReminderTimer = null;
      button.textContent = 'Start hydration reminders';
      status.textContent = 'Hydration reminders are off. Follow your clinician’s fluid advice.';
      return;
    }
    const minutes = Number($('#hydration-interval').value);
    if (![30, 60, 90].includes(minutes)) return;
    hydrationReminderTimer = window.setInterval(() => {
      status.textContent = 'A gentle reminder: consider having a drink if you are thirsty. Follow your clinician’s advice; this is not a prescribed fluid amount.';
    }, minutes * 60 * 1000);
    button.textContent = 'Stop hydration reminders';
    status.textContent = `On-page reminder is set for every ${minutes} minutes while this page stays open.`;
  }

  async function initialize() {
    initializeNavigation();
    $('#pregnancy-profile-form').addEventListener('submit', savePregnancy);
    $('#enable-notifications').addEventListener('click', enableNotifications);
    $('#disable-notifications').addEventListener('click', disableNotifications);
    $('#mood-entry-form').addEventListener('submit', saveMood);
    $('#assistant-form').addEventListener('submit', sendAssistantMessage);
    $('#assistant-conversation-select').addEventListener('change', async (event) => {
      activeConversationId = Number(event.currentTarget.value);
      await loadConversation();
    });
    $('#new-assistant-chat').addEventListener('click', async () => {
      try {
        const data = await api('/api/assistant/conversations', { method: 'POST' });
        await loadConversationList(data.conversation.id);
      } catch (error) {
        showStatus('#assistant-status', error.message, true);
      }
    });
    $('#delete-assistant-chat').addEventListener('click', async () => {
      if (!activeConversationId || !window.confirm('Permanently delete this conversation and its messages?')) return;
      try {
        await api(`/api/assistant/conversations/${activeConversationId}`, { method: 'DELETE' });
        activeConversationId = null;
        await loadConversationList();
      } catch (error) {
        showStatus('#assistant-status', error.message, true);
      }
    });
    $('#generate-meal-ideas').addEventListener('click', generateMealIdeas);
    $('#find-maternity-hospital').addEventListener('click', findNearbyHospital);
    $('#hydration-reminder-toggle').addEventListener('click', toggleHydrationReminder);
    $('#hydration-interval').addEventListener('change', () => {
      if (hydrationReminderTimer !== null) {
        toggleHydrationReminder();
        toggleHydrationReminder();
      }
    });
    $('#mood-month-previous').addEventListener('click', () => {
      calendarMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() - 1, 1);
      loadMood().catch((error) => showStatus('#mood-status', error.message, true));
    });
    $('#mood-month-next').addEventListener('click', () => {
      const next = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 1);
      const today = new Date();
      if (next.getFullYear() > today.getFullYear() || (next.getFullYear() === today.getFullYear() && next.getMonth() > today.getMonth())) return;
      calendarMonth = next;
      loadMood().catch((error) => showStatus('#mood-status', error.message, true));
    });

    const loaders = [
      ['pregnancy', loadPregnancy],
      ['notification settings', loadNotificationStatus],
      ['mood', loadMood],
      ['mental health progress', loadMentalHealth],
      ['assistant', loadConversationList],
      ['relaxation activities', loadMeditation],
      ['nutrition', loadNutrition],
      ['emergency support', loadEmergency],
    ];
    await Promise.all(loaders.map(async ([name, loader]) => {
      try {
        await loader();
      } catch (error) {
        console.error(`Could not load ${name}.`, error);
        if (name === 'pregnancy') showStatus('#pregnancy-status', error.message, true);
        if (name === 'mood') showStatus('#mood-status', error.message, true);
        if (name === 'mental health progress') $('#mental-health-notices').textContent = error.message;
        if (name === 'assistant') showStatus('#assistant-status', error.message, true);
        if (name === 'relaxation activities') showStatus('#meditation-status', error.message, true);
        if (name === 'nutrition') showStatus('#nutrition-status', error.message, true);
        if (name === 'emergency support') $('#emergency-guidance').textContent = error.message;
      }
    }));
  }

  initialize();
})();
