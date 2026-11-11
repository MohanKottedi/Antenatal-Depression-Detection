"""Time-zone-aware reminder selection and idempotent push delivery."""

import json
import logging
import os
import time
from datetime import datetime, time as local_time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database import (
    NotificationDelivery,
    PatientProfile,
    PregnancyProfile,
    PushSubscription,
)


logger = logging.getLogger(__name__)
DELIVERY_RETRY_DELAY = timedelta(minutes=5)


def due_reminders(profile: PregnancyProfile, now: datetime) -> list[tuple[str, int]]:
    """Return reminder keys due now, with the current pregnancy week when applicable."""
    if not profile.reminder_timezone:
        return []
    try:
        local_now = now.astimezone(ZoneInfo(profile.reminder_timezone))
    except (ValueError, KeyError):
        logger.error("Skipping reminders for invalid timezone on pregnancy profile %s", profile.id)
        return []

    try:
        hour_text, minute_text = profile.reminder_time.split(":")
        scheduled_time = local_time(int(hour_text), int(minute_text))
    except (AttributeError, ValueError):
        logger.error("Skipping reminders for invalid time on pregnancy profile %s", profile.id)
        return []

    if profile.lmp_date:
        pregnancy_start = profile.lmp_date
    elif profile.confirmed_due_date:
        pregnancy_start = profile.confirmed_due_date - timedelta(days=280)
    else:
        pregnancy_start = None

    due: list[tuple[str, int]] = []
    today = local_now.date()
    if profile.screening_reminders and profile.follow_up_reminder_date:
        reminder_date = profile.follow_up_reminder_date
        if reminder_date <= today and (today - reminder_date).days <= 1:
            scheduled_at = datetime.combine(
                reminder_date,
                scheduled_time,
                tzinfo=local_now.tzinfo,
            )
            if timedelta(0) <= local_now - scheduled_at < timedelta(days=1):
                due.append((f"screening:{reminder_date.isoformat()}", 0))

    if profile.weekly_tracker_reminders and pregnancy_start:
        elapsed_days = (today - pregnancy_start).days
        week = elapsed_days // 7
        if 0 < week <= 42 and elapsed_days % 7 == 0:
            scheduled_at = datetime.combine(
                today,
                scheduled_time,
                tzinfo=local_now.tzinfo,
            )
            if local_now >= scheduled_at:
                due.append(
                    (f"pregnancy-week:{pregnancy_start.isoformat()}:{week}", week)
                )
    return due


def _payload(reminder_key: str) -> str:
    destination = "pregnancy" if reminder_key.startswith("pregnancy-week:") else "mental-health"
    return json.dumps(
        {
            "title": "Sneha reminder",
            "body": "Open your private Sneha account to see your reminder.",
            "url": f"/patient#{destination}",
            "tag": reminder_key,
        }
    )


def _send_web_push(subscription: PushSubscription, payload: str) -> None:
    from pywebpush import webpush

    private_key = os.getenv("VAPID_PRIVATE_KEY")
    subject = os.getenv("VAPID_SUBJECT", "mailto:admin@example.com")
    if not private_key:
        raise RuntimeError("VAPID_PRIVATE_KEY is not configured.")
    webpush(
        subscription_info={
            "endpoint": subscription.endpoint,
            "keys": {
                "p256dh": subscription.p256dh_key,
                "auth": subscription.auth_key,
            },
        },
        data=payload,
        vapid_private_key=private_key,
        vapid_claims={"sub": subject},
    )


def deliver_due_reminders(
    db: Session,
    now: datetime | None = None,
    send_push=_send_web_push,
) -> int:
    """Send each due reminder once to each subscribed browser."""
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    profiles = (
        db.query(PregnancyProfile, PatientProfile.user_id)
        .join(PatientProfile, PregnancyProfile.patient_id == PatientProfile.id)
        .filter(PatientProfile.user_id.isnot(None))
        .all()
    )
    sent_count = 0
    for profile, user_id in profiles:
        subscriptions = db.query(PushSubscription).filter(
            PushSubscription.user_id == user_id
        ).all()
        if not subscriptions:
            continue
        for reminder_key, _week in due_reminders(profile, current_time):
            for subscription in subscriptions:
                delivery_key = f"{reminder_key}:{subscription.endpoint_hash}"
                delivery = db.query(NotificationDelivery).filter(
                    NotificationDelivery.user_id == user_id,
                    NotificationDelivery.reminder_key == delivery_key,
                ).first()
                if delivery and delivery.status == "sent":
                    continue
                if (
                    delivery
                    and current_time.replace(tzinfo=None) - delivery.last_attempt_at
                    < DELIVERY_RETRY_DELAY
                ):
                    continue

                if delivery is None:
                    delivery = NotificationDelivery(
                        user_id=user_id,
                        reminder_key=delivery_key,
                        status="pending",
                        last_attempt_at=current_time.replace(tzinfo=None),
                    )
                    db.add(delivery)
                    try:
                        db.commit()
                    except IntegrityError:
                        db.rollback()
                        continue
                else:
                    delivery.status = "pending"
                    delivery.last_attempt_at = current_time.replace(tzinfo=None)
                    delivery.last_error = None
                    db.commit()

                try:
                    send_push(subscription, _payload(reminder_key))
                    delivery.status = "sent"
                    delivery.sent_at = current_time.replace(tzinfo=None)
                    delivery.last_error = None
                    sent_count += 1
                except Exception as error:
                    status_code = getattr(
                        getattr(error, "response", None),
                        "status_code",
                        None,
                    )
                    if status_code in (404, 410):
                        db.delete(subscription)
                    logger.error(
                        "Push delivery failed for patient user %s (HTTP %s, %s)",
                        user_id,
                        status_code or "unknown",
                        type(error).__name__,
                    )
                    delivery.status = "failed"
                    delivery.sent_at = None
                    delivery.last_error = (
                        f"Push service error ({status_code})"
                        if status_code
                        else f"Push service error ({type(error).__name__})"
                    )
                db.commit()
    return sent_count


def run_worker(poll_seconds: int = 30) -> None:
    from src.database import SessionLocal

    if not os.getenv("VAPID_PUBLIC_KEY") or not os.getenv("VAPID_PRIVATE_KEY"):
        raise RuntimeError(
            "Set VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY before starting the notification worker."
        )
    logger.info("Starting Sneha push notification worker.")
    while True:
        db = SessionLocal()
        try:
            deliver_due_reminders(db)
        except Exception:
            db.rollback()
            logger.exception("Notification worker cycle failed.")
        finally:
            db.close()
        time.sleep(poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_worker()
