self.addEventListener('push', (event) => {
  let notification = {
    title: 'Sneha reminder',
    body: 'Open your private Sneha account to see your reminder.',
    url: '/patient#pregnancy',
  };
  if (event.data) {
    try {
      notification = { ...notification, ...event.data.json() };
    } catch (error) {
      console.error('Could not read the push notification payload.', error);
    }
  }
  event.waitUntil(self.registration.showNotification(notification.title, {
    body: notification.body,
    data: { url: notification.url },
    tag: notification.tag || 'sneha-reminder',
    renotify: false,
  }));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const destination = new URL(event.notification.data?.url || '/patient#pregnancy', self.location.origin);
  event.waitUntil((async () => {
    const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const windowClient of windows) {
      if (new URL(windowClient.url).origin === self.location.origin && 'focus' in windowClient) {
        await windowClient.focus();
        await windowClient.navigate(destination.href);
        return;
      }
    }
    await self.clients.openWindow(destination.href);
  })());
});
