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


st.sidebar.title("🏷️ Semantic Tagging")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Add Document",
        "Create Tag",
    ],
)


if page == "Dashboard":
    st.title("📊 Dashboard")

    st.markdown(
        "2-stage pipeline: bi-encoder retrieval → cross-encoder reranking"
    )

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

    if not docs:
        st.info("No documents for this tag")
    else:
        id_filter = st.text_input("Filter by ID", placeholder="Paste a document ID...")
        if id_filter:
            docs = [d for d in docs if id_filter.strip().lower() in d["id"].lower()]

        for d in docs:
            st.markdown(f"`{d['id']}`")
            st.markdown(
                f"<div style='border:1px solid #ddd; border-radius:6px; padding:12px; margin-bottom:12px'>"
                f"{d['content'][:400]}"
                f"</div>",
                unsafe_allow_html=True,
            )
            render_tags(d.get("tags"))


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
