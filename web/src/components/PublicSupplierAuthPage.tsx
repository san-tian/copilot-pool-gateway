import { useEffect, useMemo, useRef, useState } from "react"

import { api, type AccountProbeResult, type PublicAuthSession } from "../api"

function DeviceCodeCard({ userCode, verificationUri }: { userCode: string; verificationUri?: string }) {
  return (
    <div style={{ textAlign: "center", padding: "20px 0" }}>
      <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 16 }}>
        Open GitHub and enter this device code to connect or refresh your Copilot account.
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

export function PublicSupplierAuthPage() {
  const [session, setSession] = useState<PublicAuthSession | null>(null)
  const [started, setStarted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<{ accountName: string; created: boolean; probe?: AccountProbeResult } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => stopPolling, [])

  const startFlow = async () => {
    setLoading(true)
    setError("")
    setResult(null)
    try {
      const startedSession = await api.startPublicAuth()
      setSession(startedSession)
      setStarted(true)
      pollRef.current = setInterval(() => {
        void (async () => {
          try {
            const polled = await api.pollPublicAuth(startedSession.sessionId)
            setSession(polled)
            if (polled.status === "completed") {
              stopPolling()
              const completed = await api.completePublicAuth({ sessionId: startedSession.sessionId })
              setResult({ accountName: completed.account.name, created: completed.created, probe: completed.probe })
            }
            if (["expired", "error"].includes(polled.status)) {
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
  }

  const statusText = useMemo(() => {
    if (result) {
      if (result.probe && !result.probe.success) {
        return `Account synced as ${result.accountName}, but the premium-model probe failed: ${result.probe.error}`
      }
      return result.created ? `Connected new account: ${result.accountName}` : `Updated existing account: ${result.accountName}`
    }
    switch (session?.status) {
      case "completed":
        return "Finishing account sync and premium-model probe..."
      case "expired":
        return "This device code expired. Start again to get a fresh code."
      case "error":
        return session.error || "Authorization failed."
      case "pending":
        return "Waiting for GitHub authorization to finish..."
      default:
        return "Start GitHub authorization. Account type and premium-model availability are detected automatically."
    }
  }, [result, session])

  const statusColor = result?.probe && !result.probe.success
    ? "var(--red)"
    : result
      ? "var(--green)"
      : session?.status === "error" || session?.status === "expired"
        ? "var(--red)"
        : "var(--text-muted)"

  return (
    <div style={{ maxWidth: 520, margin: "80px auto", padding: "0 16px" }}>
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>Connect Copilot Account</h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 12 }}>
          Use this page to add a new supplier account or refresh an existing one. Account type and premium-model availability are tested automatically after GitHub authorization.
        </p>
        <div style={{ color: statusColor, fontSize: 14, marginBottom: 12 }}>
          {statusText}
        </div>
        {error && <div style={{ color: "var(--red)", fontSize: 13, marginBottom: 12 }}>{error}</div>}
        {!started && !result && (
          <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
            <button className="primary" onClick={() => void startFlow()} disabled={loading}>
              {loading ? "Starting..." : "Connect with GitHub"}
            </button>
          </div>
        )}
        {session?.userCode && !result && session.status !== "expired" && <DeviceCodeCard userCode={session.userCode} verificationUri={session.verificationUri} />}
      </div>
    </div>
  )
}
