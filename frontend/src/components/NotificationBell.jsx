import { useEffect, useRef, useState } from "react";
import api from "../services/api";
import "./NotificationBell.css";

function formatNotificationDate(value) {
  if (!value) {
    return "";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDay(value) {
  if (!value) {
    return "Recent";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Recent";
  }

  const today = new Date();

  const yesterday = new Date();
  yesterday.setDate(
    yesterday.getDate() - 1
  );

  if (
    date.toDateString() ===
    today.toDateString()
  ) {
    return "TODAY";
  }

  if (
    date.toDateString() ===
    yesterday.toDateString()
  ) {
    return "YESTERDAY";
  }

  return date
    .toLocaleDateString("en-IN", {
      weekday: "short",
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
    .toUpperCase();
}

function groupNotifications(notifications) {
  const groups = [];

  notifications.forEach((notification) => {
    const label = formatDay(
      notification.created_at
    );

    let group = groups.find(
      (item) => item.label === label
    );

    if (!group) {
      group = {
        label,
        items: [],
      };

      groups.push(group);
    }

    group.items.push(notification);
  });

  return groups;
}

function NotificationIcon({
  level,
}) {
  if (
    level === "high" ||
    level === "critical"
  ) {
    return (
      <span className="live-notification-icon alert">
        !
      </span>
    );
  }

  if (level === "medium") {
    return (
      <span className="live-notification-icon warning">
        !
      </span>
    );
  }

  return (
    <span className="live-notification-icon info">
      i
    </span>
  );
}

export default function NotificationBell({
  variant = "default",
}) {
  const [open, setOpen] =
    useState(false);

  const [
    notifications,
    setNotifications,
  ] = useState([]);

  const [
    unreadCount,
    setUnreadCount,
  ] = useState(0);

  const [
    loading,
    setLoading,
  ] = useState(false);

  const wrapperRef = useRef(null);

  const loadNotifications = async () => {
    try {
      const response =
        await api.get(
          "/notifications"
        );

      const data =
        response.data || {};

      setNotifications(
        Array.isArray(
          data.notifications
        )
          ? data.notifications
          : []
      );

      setUnreadCount(
        Number(
          data.unread_count || 0
        )
      );
    } catch (error) {
      console.error(
        "Failed to load notifications:",
        error
      );
    }
  };

  useEffect(() => {
    loadNotifications();

    const interval = setInterval(
      loadNotifications,
      10000
    );

    return () => {
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const handleOutsideClick =
      (event) => {
        if (
          wrapperRef.current &&
          !wrapperRef.current.contains(
            event.target
          )
        ) {
          setOpen(false);
        }
      };

    document.addEventListener(
      "mousedown",
      handleOutsideClick
    );

    return () => {
      document.removeEventListener(
        "mousedown",
        handleOutsideClick
      );
    };
  }, []);

  const handleToggle = async () => {
    const nextOpen = !open;

    setOpen(nextOpen);

    if (nextOpen) {
      setLoading(true);

      await loadNotifications();

      setLoading(false);
    }
  };

  const markAllAsRead = async () => {
    try {
      await api.post(
        "/notifications/read-all"
      );

      setNotifications(
        (previous) =>
          previous.map(
            (notification) => ({
              ...notification,
              read: true,
            })
          )
      );

      setUnreadCount(0);
    } catch (error) {
      console.error(
        "Failed to mark notifications as read:",
        error
      );
    }
  };

  const markOneAsRead = async (
    notification
  ) => {
    if (notification.read) {
      return;
    }

    try {
      await api.post(
        `/notifications/${encodeURIComponent(
          notification.id
        )}/read`
      );

      setNotifications(
        (previous) =>
          previous.map(
            (item) =>
              item.id ===
              notification.id
                ? {
                    ...item,
                    read: true,
                  }
                : item
          )
      );

      setUnreadCount(
        (previous) =>
          Math.max(
            0,
            previous - 1
          )
      );
    } catch (error) {
      console.error(
        "Failed to mark notification as read:",
        error
      );
    }
  };

  const groups =
    groupNotifications(
      notifications
    );

  return (
    <div
      ref={wrapperRef}
      className={`live-notification-wrapper ${variant}`}
    >
      <button
        type="button"
        className="live-notification-button"
        aria-label="Notifications"
        aria-expanded={open}
        onClick={handleToggle}
      >
        <svg
          className="live-notification-bell"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
          <path d="M10 21h4" />
        </svg>

        {unreadCount > 0 && (
          <>
            <span className="live-notification-dot" />

            <span className="live-notification-count">
              {unreadCount > 9
                ? "9+"
                : unreadCount}
            </span>
          </>
        )}
      </button>

      {open && (
        <div className="live-notification-panel">
          <div className="live-notification-header">
            <div>
              <h3>
                Notifications{" "}
                {unreadCount > 0 && (
                  <span>
                    {unreadCount}
                  </span>
                )}
              </h3>

              <p>
                Analysis and follow-up alerts
              </p>
            </div>

            <button
              type="button"
              className="live-notification-mark"
              onClick={
                markAllAsRead
              }
              disabled={
                unreadCount === 0
              }
            >
              Mark All As Read
            </button>
          </div>

          {loading ? (
            <div className="live-notification-empty">
              <div className="live-notification-spinner"></div>
              <p>
                Loading notifications...
              </p>
            </div>
          ) : notifications.length ===
            0 ? (
            <div className="live-notification-empty">
              <span>✓</span>
              <p>
                No notifications.
              </p>
            </div>
          ) : (
            <div className="live-notification-scroll">
              {groups.map(
                (group) => (
                  <div
                    key={
                      group.label
                    }
                  >
                    <div className="live-notification-date">
                      {group.label}
                    </div>

                    {group.items.map(
                      (
                        notification
                      ) => (
                        <button
                          type="button"
                          key={
                            notification.id
                          }
                          className={`live-notification-item ${
                            notification.read
                              ? "read"
                              : "unread"
                          }`}
                          onClick={() =>
                            markOneAsRead(
                              notification
                            )
                          }
                        >
                          <NotificationIcon
                            level={
                              notification.level
                            }
                          />

                          <span className="live-notification-content">
                            <span className="live-notification-title">
                              {
                                notification.title
                              }
                            </span>

                            <span className="live-notification-message">
                              {
                                notification.message
                              }
                            </span>

                            <span className="live-notification-time">
                              {notification.created_at
                                ? `${formatNotificationDate(
                                    notification.created_at
                                  )}`
                                : ""}
                            </span>
                          </span>

                          {!notification.read && (
                            <span className="live-notification-unread-dot" />
                          )}
                        </button>
                      )
                    )}
                  </div>
                )
              )}
            </div>
          )}

          <div className="live-notification-footer">
            AI alerts are informational and
            should not be interpreted as
            confirmed medical diagnoses.
          </div>
        </div>
      )}
    </div>
  );
}