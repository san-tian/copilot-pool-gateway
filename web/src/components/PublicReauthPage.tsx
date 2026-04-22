import { useEffect, useMemo, useRef, useState } from "react"

import { api, type PublicReauthSession } from "../api"

function DeviceCodeCard({ userCode, verificationUri }: { userCode: string; verificationUri?: string }) {
  return (
    <div style={{ textAlign: "center", padding: "20px 0" }}>
      <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 16 }}>
        Open GitHub and enter this device code to refresh your Copilot login.
      </p>
      <div
        onClick={() => void navigator.clipboard.writeText(userCode)}
        style={{
          display: "inline-block",
          padding: "12px 24px",
          background: "var(--bg)",
          border: "2px solid var(--accent)",
          borderRadius: "var(--radius)",
          fontSize: 28,
          fontWeight: 700,
          fontFamily: "monospace",
          letterSpacing: 4,
          cursor: "pointer",
          userSelect: "all",
          marginBottom: 8,
        }}
        title="Click to copy"
      >
        {userCode}
      </div>
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16 }}>Click the code to copy it.</p>
      {verificationUri && (
        <a
          href={verificationUri}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: "inline-block",
            padding: "8px 20px",
            background: "var(--accent)",
            color: "#fff",
            borderRadius: "var(--radius)",
            textDecoration: "none",
            fontSize: 14,
          }}
        >
          Open GitHub verification page
        </a>
      )}
    </div>
  )
}

export function PublicReauthPage({ sessionId }: { sessionId: string }) {
  const [session, setSession] = useState<PublicReauthSession | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => stopPolling, [])

  useEffect(() => {
    void (async () => {
      try {
        const initial = await api.getPublicReauthSession(sessionId)
        setSession(initial)
        if (initial.status === "completed" || initial.status === "expired") {
          return
        }
        const started = await api.startPublicReauthSession(sessionId)
        setSession(started)
        pollRef.current = setInterval(() => {
          void (async () => {
            try {
              const polled = await api.pollPublicReauthSession(sessionId)
              setSession(polled)
              if (["completed", "expired", "error"].includes(polled.status)) {
                stopPolling()
              }
            } catch (err) {
              setError((err as Error).message)
              stopPolling()
            }
          })()
        }, 3000)
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    })()
  }, [sessionId])

  const statusText = useMemo(() => {
    switch (session?.status) {
      case "completed":
        return "GitHub authorization refreshed successfully."
      case "expired":
        return "This reauth link has expired. Ask the operator for a new link."
      case "error":
        return session.error || "Authorization failed."
      case "pending":
        return "Waiting for GitHub authorization to finish..."
      default:
        return "Preparing GitHub device authorization..."
    }
  }, [session])

  return (
    <div style={{ maxWidth: 520, margin: "80px auto", padding: "0 16px" }}>
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>Refresh Copilot Login</h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 12 }}>
          Supplier session for <strong>{session?.accountName ?? "your account"}</strong>
        </p>
        {loading && <div style={{ color: "var(--text-muted)", padding: "16px 0" }}>Loading...</div>}
        {error && <div style={{ color: "var(--red)", fontSize: 13, marginBottom: 12 }}>{error}</div>}
        {!loading && !error && (
          <>
            <div style={{ color: session?.status === "completed" ? "var(--green)" : session?.status === "error" || session?.status === "expired" ? "var(--red)" : "var(--text-muted)", fontSize: 14, marginBottom: 12 }}>
              {statusText}
            </div>
            {session?.userCode && session.status !== "completed" && session.status !== "expired" && (
              <DeviceCodeCard userCode={session.userCode} verificationUri={session.verificationUri} />
            )}
            {session?.completedAt && (
              <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
                Completed at {new Date(session.completedAt).toLocaleString()}
              </div>
            )}
            {session?.expiresAt && session.status !== "completed" && (
              <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
                Link expires at {new Date(session.expiresAt).toLocaleString()}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
