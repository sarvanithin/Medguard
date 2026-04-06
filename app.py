"""
MedGuard — Clinical LLM Safety Dashboard
Health Universe deployment — calls the live Render API.
"""

import requests
import streamlit as st

API = "https://medguard-ru03.onrender.com"
HEADERS = {"Content-Type": "application/json"}

st.set_page_config(
    page_title="MedGuard — Clinical LLM Safety",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ MedGuard")
st.caption("Healthcare-specific LLM guardrails — PHI detection, drug interaction checks & clinical safety")

# ── Health banner ──────────────────────────────────────────────────────────────
try:
    h = requests.get(f"{API}/v1/health", timeout=8).json()
    st.success(f"Service: **{h.get('status', 'unknown')}** · version {h.get('version', '—')}")
except Exception:
    st.warning("⚠️ Could not reach MedGuard API — it may be waking up (free tier). Refresh in 30s.")

st.divider()

tab1, tab2, tab3 = st.tabs(["💬 Clinical Chat", "🔍 PHI Detector", "💊 Drug Interactions"])

# ── Tab 1: Clinical Chat ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Clinical Chat with Safety Guardrails")
    st.caption("Every response is screened for PHI, drug contraindications, and clinical scope violations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a clinical question…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("MedGuard processing…"):
                try:
                    payload = {
                        "messages": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    }
                    resp = requests.post(
                        f"{API}/v1/chat",
                        json=payload,
                        headers=HEADERS,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        reply = data.get("content", "")
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})

                        guardrails = data.get("guardrail_annotations", [])
                        if guardrails:
                            with st.expander(f"🛡️ {len(guardrails)} guardrail(s) triggered"):
                                for g in guardrails:
                                    color = {"warning": "🟡", "block": "🔴", "info": "🔵"}.get(g.get("level", ""), "⚪")
                                    st.markdown(f"{color} **{g.get('type','')}** — {g.get('message','')}")
                    elif resp.status_code == 402:
                        st.error("402 Payment Required — API token needed for programmatic access.")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text[:300]}")
                except requests.exceptions.Timeout:
                    st.warning("Request timed out — service may be starting up. Try again in 30s.")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Tab 2: PHI Detector ────────────────────────────────────────────────────────
with tab2:
    st.subheader("PHI (Protected Health Information) Detector")
    st.caption("Scans clinical text for names, SSNs, dates, MRNs, addresses and other identifiers.")

    phi_text = st.text_area(
        "Paste clinical text to scan",
        height=180,
        placeholder="e.g. Patient John Smith, DOB 01/15/1980, MRN 4829301, presented to ED on 3/12…",
    )

    if st.button("Scan for PHI", type="primary"):
        if not phi_text.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Scanning…"):
                try:
                    resp = requests.post(
                        f"{API}/v1/check/phi",
                        json={"text": phi_text},
                        headers=HEADERS,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        matches = data.get("matches", [])
                        if matches:
                            st.error(f"⚠️ Found **{len(matches)} PHI item(s)**")
                            for m in matches:
                                st.markdown(f"- `{m.get('text','')}` — **{m.get('phi_type','')}**")
                            if data.get("redacted_text"):
                                st.subheader("Redacted version")
                                st.code(data["redacted_text"])
                        else:
                            st.success("✅ No PHI detected.")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text[:300]}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 3: Drug Interactions ───────────────────────────────────────────────────
with tab3:
    st.subheader("Drug Interaction & Contraindication Checker")
    st.caption("Checks against RxNorm/OpenFDA for known interactions and contraindications.")

    col1, col2 = st.columns(2)
    with col1:
        drugs_input = st.text_area(
            "Medications (one per line)",
            height=140,
            placeholder="Warfarin\nAspirin\nMetformin",
        )
    with col2:
        conditions_input = st.text_area(
            "Patient conditions (one per line, optional)",
            height=140,
            placeholder="Type 2 Diabetes\nAtrial Fibrillation",
        )

    if st.button("Check Interactions", type="primary"):
        drugs = [d.strip() for d in drugs_input.splitlines() if d.strip()]
        conditions = [c.strip() for c in conditions_input.splitlines() if c.strip()]
        if not drugs:
            st.warning("Enter at least one medication.")
        else:
            with st.spinner("Checking…"):
                try:
                    resp = requests.post(
                        f"{API}/v1/check/drug-interactions",
                        json={"drug_names": drugs, "conditions": conditions},
                        headers=HEADERS,
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        interactions = data.get("interactions", [])
                        contraindications = data.get("contraindications", [])
                        if not interactions and not contraindications:
                            st.success("✅ No known interactions or contraindications found.")
                        if interactions:
                            st.warning(f"⚠️ **{len(interactions)} interaction(s)** found")
                            for i in interactions:
                                st.markdown(f"- **{i.get('drug_a','')} + {i.get('drug_b','')}**: {i.get('description','')}")
                        if contraindications:
                            st.error(f"🚫 **{len(contraindications)} contraindication(s)** found")
                            for c in contraindications:
                                st.markdown(f"- **{c.get('drug','')}**: {c.get('reason','')}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text[:300]}")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption("Powered by [MedGuard](https://medguard-ru03.onrender.com/docs) · API docs · x402/MPP payment protocol")
