import streamlit as st
import requests
import os

API_BASE = os.getenv("BACKEND_URL", "http://backend:8000") + "/api/v1"

st.set_page_config(
    page_title="Semantic Tagging Platform",
    page_icon="🏷️",
    layout="wide",
)


def request(method, path, **kwargs):
    try:
        r = requests.request(
            method,
            f"{API_BASE}{path}",
            timeout=60,
            **kwargs
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(str(e))
        return None


def get(path, params=None):
    return request("GET", path, params=params)


def post(path, payload=None):
    return request("POST", path, json=payload)


def format_conf(score):
    return f"{score:.0%}" if score is not None else "0%"


def render_tags(tags):
    if not tags:
        st.caption("No tags")
        return
    for t in tags:
        name = t.get("name", "?")
        conf = t.get("confidence", 0)
        verified = "✓" if t.get("is_human_verified") else ""
        st.write(f"- **{name}** {verified} ({format_conf(conf)})")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("🏷️ Semantic Tagging")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Add Document",
        "Create Tag",
        "Explore Tags",
        "Review Queue",
        "Fine-tune Model",
    ],
)

stats = get("/documents/stats/summary")
if stats:
    st.sidebar.divider()
    st.sidebar.metric("Documents", stats.get("documents", 0))
    st.sidebar.metric("Tags", stats.get("tags", 0))


# ═════════════════════════════════════════════
# 1. DASHBOARD
# ═════════════════════════════════════════════

if page == "Dashboard":
    st.title("📊 Dashboard")

    st.markdown(
        "2-stage pipeline: bi-encoder retrieval → cross-encoder reranking"
    )

    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Documents", stats["documents"])
        c2.metric("Tags", stats["tags"])

    st.divider()

    st.subheader("Recent Documents")
    docs = get("/documents/", {"limit": 10})

    if docs:
        for d in docs:
            with st.expander(d.get("title") or d["id"][:12]):
                st.write(d["content"][:400])
                render_tags(d.get("tags"))
                if d.get("dbpedia_label"):
                    st.caption(f"Ground truth: {d['dbpedia_label']}")
    else:
        st.info("No documents yet.")


# ═════════════════════════════════════════════
# 2. ADD DOCUMENT
# ═════════════════════════════════════════════

elif page == "Add Document":
    st.title("📄 Add Document")

    title = st.text_input("Title")
    content = st.text_area("Content", height=200)

    if st.button("Ingest & Tag"):
        if not content:
            st.warning("Content required")
        else:
            res = post("/documents/", {
                "title": title,
                "content": content,
            })

            if res:
                st.success("Document processed")
                st.subheader("Assigned Tags")
                render_tags(res.get("tags"))


# ═════════════════════════════════════════════
# 3. CREATE TAG
# ═════════════════════════════════════════════

elif page == "Create Tag":
    st.title("🏷️ Create Tag")

    name = st.text_input("Tag name")
    desc = st.text_area("Description")

    if name and len(name) > 1:
        similar = get("/tags/search", {"q": name, "top_k": 5})

        if similar:
            st.subheader("Similar tags")
            for s in similar:
                st.write(f"- **{s['tag']['name']}**")

    col1, col2 = st.columns(2)

    if col1.button("Create"):
        res = post("/tags/", {
            "name": name,
            "description": desc,
            "force_create": False,
        })

        if res:
            if res["created"]:
                st.success(f"Created: {res['tag']['name']}")
            else:
                st.warning(f"Reused existing tag: {res['tag']['name']}")

    if col2.button("Force create"):
        res = post("/tags/", {
            "name": name,
            "description": desc,
            "force_create": True,
        })
        if res:
            st.success(f"Created: {res['tag']['name']}")


# ═════════════════════════════════════════════
# 4. EXPLORE TAGS
# ═════════════════════════════════════════════

elif page == "Explore Tags":
    st.title("🔍 Explore Tags")

    tags = get("/tags/", {"limit": 200})

    if not tags:
        st.info("No tags")
        st.stop()

    tag_map = {t["name"]: t["id"] for t in tags}
    selected = st.selectbox("Select tag", list(tag_map.keys()))

    tag_id = tag_map[selected]

    tag = get(f"/tags/{tag_id}")
    if tag:
        st.subheader(tag["name"])
        st.caption(tag.get("description", ""))

    st.divider()
    st.subheader("Documents")

    docs = get(f"/documents/by-tag/{tag_id}", {"limit": 50})

    if docs:
        for d in docs:
            with st.expander(d.get("title") or d["id"][:12]):
                st.write(d["content"][:400])
                render_tags(d.get("tags"))
    else:
        st.info("No documents for this tag")


# ═════════════════════════════════════════════
# 5. REVIEW QUEUE
# ═════════════════════════════════════════════

elif page == "Review Queue":
    st.title("🧑‍⚖️ Review Queue")

    reviews = get("/documents/reviews/pending", {"limit": 20})

    if not reviews:
        st.success("No pending reviews")
        st.stop()

    for r in reviews:
        with st.expander(r["document_id"][:12], expanded=True):
            st.write(r["document_content"][:300])

            options = {
                c["tag_name"]: c["tag_id"]
                for c in r.get("candidates", [])
            }

            selected = st.selectbox(
                "Select tag",
                list(options.keys()),
                key=r["review_id"],
            )

            new_tag = st.text_input(
                "Or new tag",
                key=f"new_{r['review_id']}",
            )

            if st.button("Resolve", key=r["review_id"]):
                payload = {
                    "review_id": r["review_id"],
                    "accepted_tag_ids": [options[selected]] if selected else [],
                    "new_tag_name": new_tag or None,
                }

                res = post("/documents/reviews/resolve", payload)
                if res:
                    st.success("Resolved")
                    st.rerun()
