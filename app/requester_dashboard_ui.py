REQUESTER_DASHBOARD_STYLES = """
    .table-scroll { overflow-x: auto; }
    .requester-label { min-width: 210px; }
    .requester-signals { min-width: 220px; color: var(--muted); font-size: 12px; }
    .requester-client-mix { min-width: 150px; color: var(--muted); font-size: 12px; }
    .requester-robots { min-width: 130px; color: var(--muted); font-size: 12px; }
    .risk-pill.normal { color: var(--good); background: rgba(17, 122, 101, 0.10); }
    .risk-pill.watch { color: var(--warm); background: rgba(217, 130, 43, 0.12); }
    .risk-pill.high { color: var(--danger); background: rgba(187, 45, 59, 0.10); }
"""


REQUESTER_DASHBOARD_MARKUP = """
      <div class="panel card span-12">
        <div class="label">Traffic Attribution</div>
        <h2>Requester Usage</h2>
        <p class="muted">HF accounts are resolved asynchronously. Token, network, and reported robot identifiers are one-way fingerprints; raw tokens, IP addresses, and hardware IDs are never stored. Robot IDs are client-reported telemetry, not hardware attestation.</p>
        <div class="fleet-summary" id="requester-summary"></div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Requester</th>
                <th>Status</th>
                <th>Requests</th>
                <th>Allocated</th>
                <th>Connected</th>
                <th>Traffic</th>
                <th>Peak</th>
                <th>Networks</th>
                <th>Reported robots</th>
                <th>Clients</th>
                <th>Signals</th>
              </tr>
            </thead>
            <tbody id="requester-leaderboard"></tbody>
          </table>
        </div>
        <div class="footer-note" id="requester-detail"></div>
      </div>
"""


REQUESTER_DASHBOARD_KPI_CARDS = """
        kpiCard(`HF users / ${windowLabel}`, prettyNumber(summary.authenticated_users_window || 0), `Distinct verified Hugging Face accounts in the last ${windowLabel}`),
        kpiCard(`Connected requesters / ${windowLabel}`, prettyNumber(summary.connected_requesters_window || 0), `Distinct requesters whose allocated session reached the compute websocket`),
        kpiCard(`Anonymous IPs / ${windowLabel}`, prettyNumber(summary.anonymous_ips_window || 0), `Distinct privacy-safe network fingerprints without tokens`),
        kpiCard(`Reported robots / ${windowLabel}`, prettyNumber(summary.reported_robots_window || 0), `Distinct privacy-safe robot fingerprints reported by clients; not verified hardware`),
        kpiCard(`Flagged / ${windowLabel}`, prettyNumber(summary.unusual_requesters_window || 0), `Requesters with volume, burst, network, token, or automation signals`),
"""


REQUESTER_DASHBOARD_SCRIPT = """
    function requesterStatusClass(row) {
      if (row.risk === 'high') return 'bad';
      if (row.risk === 'watch' || row.verification === 'pending' || row.verification === 'unavailable') return 'warm';
      return 'good';
    }

    function requesterStatusLabel(row) {
      const labels = {
        verified: 'verified HF',
        pending: 'verifying',
        unavailable: 'lookup unavailable',
        invalid: 'invalid token',
        unrecognized: 'unrecognized token',
        not_provided: 'no token',
        not_applicable: 'other',
      };
      return labels[row.verification] || row.kind || 'unknown';
    }

    function requesterClientMix(clientKinds) {
      const entries = Object.entries(clientKinds || {}).slice(0, 3);
      if (!entries.length) return 'unknown';
      return entries.map(([kind, count]) => `${kind.replace('automation:', '')}: ${prettyNumber(count)}`).join(' · ');
    }

    function requesterRobotMix(row) {
      const count = Number(row.reported_robot_count || 0);
      if (!count) return 'not reported';
      const fingerprints = (row.reported_robot_ids || []).slice(0, 2).map((value) => {
        const fingerprint = String(value).replace(/^robot:/, '').slice(0, 8);
        return `•${fingerprint}`;
      });
      const countLabel = `${prettyNumber(count)}${row.reported_robot_count_overflow ? '+' : ''}`;
      return fingerprints.length ? `${countLabel} · ${fingerprints.join(', ')}` : countLabel;
    }

    function renderRequesterUsage(requesters, summary) {
      const rows = requesters?.leaderboard || [];
      const windowLabel = summary.window_label || '6h';
      document.getElementById('requester-summary').innerHTML = [
        `<span class="status-pill good">${htmlEscape(prettyNumber(summary.authenticated_users_window || 0))} HF users</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.tokens_window || 0))} tokens</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.allocated_requesters_window || 0))} allocated · ${htmlEscape(prettyNumber(summary.connected_requesters_window || 0))} connected</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.anonymous_ips_window || 0))} anonymous IPs</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.reported_robots_window || 0))} reported robots</span>`,
        `<span class="status-pill ${summary.unusual_requesters_window ? 'bad' : 'good'}">${htmlEscape(prettyNumber(summary.unusual_requesters_window || 0))} flagged</span>`,
      ].join('');

      document.getElementById('requester-leaderboard').innerHTML = rows.length ? rows.map((row) => {
        const statusClass = requesterStatusClass(row);
        const networks = `${prettyNumber(row.network_count || 0)}${row.network_count_overflow ? '+' : ''}`;
        const signals = (row.signals || []).join(' · ') || 'No unusual signal';
        return `
          <tr>
            <td class="requester-label">
              <div><strong>${htmlEscape(row.label || 'Unknown requester')}</strong></div>
              <div class="muted mono" style="margin-top:4px;">${htmlEscape(row.actor_id || '')}</div>
            </td>
            <td>
              <span class="status-pill ${statusClass}">${htmlEscape(requesterStatusLabel(row))}</span>
              <div style="margin-top:6px;"><span class="tiny-pill risk-pill ${htmlEscape(row.risk || 'normal')}">${htmlEscape(row.risk || 'normal')}</span></div>
            </td>
            <td><strong>${htmlEscape(prettyNumber(row.requests || 0))}</strong><div class="muted">${htmlEscape(prettyNumber(row.requests_per_hour || 0))}/h</div></td>
            <td>${htmlEscape(prettyNumber(row.successes || 0))}<div class="muted">${htmlEscape(row.success_rate_pct || 0)}%</div></td>
            <td><strong>${htmlEscape(prettyNumber(row.connections || 0))}</strong><div class="muted">websocket joins</div></td>
            <td>${htmlEscape(row.traffic_share_pct || 0)}%</td>
            <td>${htmlEscape(prettyNumber(row.peak_requests_per_minute || 0))}/min</td>
            <td>${htmlEscape(networks)}</td>
            <td class="requester-robots">${htmlEscape(requesterRobotMix(row))}<div class="muted">${htmlEscape(prettyNumber(row.reported_robot_requests || 0))} requests</div></td>
            <td class="requester-client-mix">${htmlEscape(requesterClientMix(row.client_kinds))}</td>
            <td class="requester-signals">${htmlEscape(signals)}</td>
          </tr>
        `;
      }).join('') : '<tr><td colspan="11" class="muted">No attributed session requests in this window yet.</td></tr>';

      const unattributed = Number(requesters?.unattributed_requests || 0);
      const attributionDetail = unattributed
        ? `${prettyNumber(unattributed)} request(s) in the last ${windowLabel} predate attribution or could not be attributed.`
        : `All recorded session requests in the last ${windowLabel} have requester attribution.`;
      document.getElementById('requester-detail').textContent = `${attributionDetail} Connected counts the first compute websocket callback for an allocated session. Allocated and Connected are independent event counts in the selected window, not a cohort conversion rate.`;
    }
"""


def inject_requester_dashboard(html: str) -> str:
    replacements = {
        "__REQUESTER_DASHBOARD_STYLES__": REQUESTER_DASHBOARD_STYLES.strip("\n"),
        "__REQUESTER_DASHBOARD_MARKUP__": REQUESTER_DASHBOARD_MARKUP.strip("\n"),
        "__REQUESTER_DASHBOARD_KPI_CARDS__": REQUESTER_DASHBOARD_KPI_CARDS.strip("\n"),
        "__REQUESTER_DASHBOARD_SCRIPT__": REQUESTER_DASHBOARD_SCRIPT.strip("\n") + "\n",
    }
    for marker, content in replacements.items():
        html = html.replace(marker, content)
    return html
