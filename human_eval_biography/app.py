import html
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import streamlit as st
import streamlit.components.v1 as components
from botocore.exceptions import ClientError


st.set_page_config(
    page_title="Claim Extraction Annotation",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [class*="_profileContainer_"] {display: none !important;}
    [class*="_profilePreview_"] {display: none !important;}
    [class*="_viewerBadge_"] {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}
    img[data-testid="appCreatorAvatar"] {display: none !important;}
    a[href*="share.streamlit.io/user/"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Le badge profil est injecté dans la page parente (hors du DOM de st.markdown).
# components.html crée un vrai iframe dont les scripts s'exécutent,
# et window.parent permet d'accéder au document parent (même origine).
components.html(
    """
    <script>
    (function() {
        function hide(doc) {
            [
                '[data-testid="appCreatorAvatar"]',
                '[class*="_profileContainer_"]',
                '[class*="_profilePreview_"]',
                'a[href*="share.streamlit.io/user/"]',
            ].forEach(function(sel) {
                doc.querySelectorAll(sel).forEach(function(el) {
                    var container = el;
                    for (var i = 0; i < 6; i++) {
                        container.style.setProperty('display', 'none', 'important');
                        if (!container.parentElement) break;
                        container = container.parentElement;
                        if (container.className && typeof container.className === 'string'
                                && container.className.indexOf('_profileContainer_') !== -1) {
                            container.style.setProperty('display', 'none', 'important');
                            break;
                        }
                    }
                });
            });
        }
        function run() {
            try { hide(window.parent.document); } catch(e) {}
            try { hide(document); } catch(e) {}
        }
        run();
        try {
            new MutationObserver(run).observe(
                window.parent.document.documentElement,
                {childList: true, subtree: true}
            );
        } catch(e) {}
    })();
    </script>
    """,
    height=0,
)


ANNOTATION_OPTIONS = ["<Unlabeled>", "True", "Not a claim", "Almost"]
CLAIM_FIELDS = ["subject", "predicate", "object", "time", "location", "reason", "manner", "hedge"]
EDITABLE_CLAIM_FIELDS = ["claim"] + CLAIM_FIELDS
EDIT_WIDGET_VERSION = "v2"
ANNOTATOR_CONFIGS = {
    "English": {"route": "en", "target_lang": None, "country_group": None},
    "Chinese": {"route": "target", "target_lang": "zh", "country_group": "China"},
    "French": {"route": "target", "target_lang": "fr", "country_group": "France"},
    "Azerbaijani": {"route": "target", "target_lang": "az", "country_group": "Azerbaijan"},
}
ANNOTATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{3,20}$")


@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "us-east-1"),
    )


def s3_bucket() -> str:
    return st.secrets["S3_BUCKET_NAME"]


def s3_object_exists(key: str) -> bool:
    try:
        get_s3_client().head_object(Bucket=s3_bucket(), Key=key)
        return True
    except ClientError:
        return False


def s3_read_json(key: str) -> dict[str, Any]:
    try:
        response = get_s3_client().get_object(Bucket=s3_bucket(), Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError:
        return {}


def s3_write_text(key: str, content: str) -> None:
    get_s3_client().put_object(
        Bucket=s3_bucket(),
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="application/json",
    )


def discover_jsonl_files(base_dir: Path) -> list[Path]:
    files = []
    for path in base_dir.rglob("*.jsonl"):
        if "annotations" in path.parts:
            continue
        files.append(path.resolve())
    return sorted(files)


@st.cache_data(show_spinner=False)
def load_jsonl_records(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def user_annotation_s3_keys(language_code: str, annotator_id: str) -> tuple[str, str]:
    stem = f"claw_4_sent_annotations_{language_code}_by_{annotator_id}"
    return (
        f"annotations/{stem}.json",
        f"annotations/{stem}.jsonl",
    )


def user_meta_s3_key(language_code: str, annotator_id: str) -> str:
    stem = f"claw_4_sent_meta_{language_code}_by_{annotator_id}"
    return f"annotations/meta_files/{stem}.json"


def load_saved_annotations(s3_key: str) -> dict[str, dict[str, Any]]:
    return s3_read_json(s3_key)


def load_meta_file(s3_key: str) -> dict[str, Any]:
    return s3_read_json(s3_key)


def write_annotation_files(
    annotation_store: dict[str, dict[str, Any]],
    json_key: str,
    jsonl_key: str,
) -> None:
    s3_write_text(json_key, json.dumps(annotation_store, ensure_ascii=False, indent=2, sort_keys=True))
    lines = [
        json.dumps(annotation_store[sample_id], ensure_ascii=False)
        for sample_id in sorted(
            annotation_store,
            key=lambda key: (
                annotation_store[key].get("qid", ""),
                annotation_store[key].get("sent_idx", -1),
            ),
        )
    ]
    s3_write_text(jsonl_key, "\n".join(lines) + ("\n" if lines else ""))


def write_meta_file(meta_store: dict[str, Any], meta_key: str) -> None:
    s3_write_text(meta_key, json.dumps(meta_store, ensure_ascii=False, indent=2, sort_keys=True))


def existing_registered_sessions() -> dict[str, dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    prefix = "annotations/meta_files/claw_4_sent_meta_"
    try:
        paginator = get_s3_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s3_bucket(), Prefix=prefix):
            for obj in page.get("Contents", []):
                try:
                    data = s3_read_json(obj["Key"])
                    annotator_id = data.get("annotator_id")
                    if annotator_id:
                        sessions[annotator_id] = data
                except Exception:
                    continue
    except ClientError:
        pass
    return sessions


def sample_id_for(record: dict[str, Any]) -> str:
    return f"{record.get('qid', 'unknown')}__{record.get('sent_idx', 'unknown')}"


def session_sample_id_for(dataset_id: str, record: dict[str, Any]) -> str:
    return f"{dataset_id}::{sample_id_for(record)}"


def saved_sample_ids(records: list[dict[str, Any]], annotation_store: dict[str, dict[str, Any]]) -> set[str]:
    valid_ids = {sample_id_for(record) for record in records}
    return {sample_id for sample_id in annotation_store if sample_id in valid_ids}


def next_unsaved_index(records: list[dict[str, Any]], annotation_store: dict[str, dict[str, Any]]) -> int | None:
    saved_ids = saved_sample_ids(records, annotation_store)
    for index, record in enumerate(records):
        if sample_id_for(record) not in saved_ids:
            return index
    return None


def humanize_model_name(raw_model: str) -> str:
    parts = [part.upper() if part.lower() == "gpt" else part for part in raw_model.split("_") if part]
    if not parts:
        return raw_model
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}-{parts[1]}" + ("." + ".".join(parts[2:]) if len(parts) > 2 else "")


def describe_data_file(path: Path) -> str:
    stem = path.stem
    match = re.match(r"human_val_annotation_sentences_(en|target)_(.+)", stem, re.IGNORECASE)
    if not match:
        return str(path.relative_to(Path.cwd()))
    source_lang = match.group(1).upper()
    model = humanize_model_name(match.group(2))
    short_name = path.name
    return f"{source_lang} | {model} | {short_name}"


def source_file_groups(jsonl_files: list[Path]) -> dict[str, list[Path]]:
    grouped = {"en": [], "target": []}
    for path in jsonl_files:
        stem = path.stem.lower()
        if "_sentences_en_" in stem:
            grouped["en"].append(path)
        elif "_sentences_target_" in stem:
            grouped["target"].append(path)
    return grouped


def resolve_source_files(jsonl_files: list[Path]) -> dict[str, Path]:
    grouped = source_file_groups(jsonl_files)
    resolved: dict[str, Path] = {}
    for key, paths in grouped.items():
        if paths:
            resolved[key] = sorted(paths)[0]
    return resolved


def annotator_selection_key() -> str:
    return "_annotator_language"


def annotator_id_key() -> str:
    return "_annotator_id"


def annotator_mode_key() -> str:
    return "_annotator_mode"


def active_time_sample_key(language_code: str, annotator_id: str) -> str:
    return f"_active-time-sample::{language_code}::{annotator_id}"


def active_time_started_at_key(language_code: str, annotator_id: str) -> str:
    return f"_active-time-start::{language_code}::{annotator_id}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def language_code_for_annotator(annotator_language: str) -> str:
    return {
        "English": "en",
        "Chinese": "zh",
        "French": "fr",
        "Azerbaijani": "az",
    }[annotator_language]


def validate_annotator_id(value: str) -> str | None:
    if not value.strip():
        return "Annotator ID is required."
    if not ANNOTATOR_ID_PATTERN.fullmatch(value.strip()):
        return "Annotator ID must be 3-20 characters and contain only letters or numbers."
    return None


def existing_annotator_ids_for_language(language_code: str) -> set[str]:
    sessions = existing_registered_sessions()
    return {
        annotator_id
        for annotator_id, data in sessions.items()
        if data.get("language_code") == language_code
    }


def build_initial_meta_store(
    annotator_id: str,
    annotator_language: str,
    language_code: str,
    data_path: Path,
) -> dict[str, Any]:
    return {
        "annotator_id": annotator_id,
        "annotator_language": annotator_language,
        "language_code": language_code,
        "created_at": now_iso(),
        "last_opened_at": now_iso(),
        "last_saved_at": None,
        "source_file": data_path.name,
        "time_by_sample": {},
    }


def filtered_records_for_annotator(records: list[dict[str, Any]], annotator_language: str) -> list[dict[str, Any]]:
    config = ANNOTATOR_CONFIGS[annotator_language]
    if config["route"] == "en":
        return records
    return [record for record in records if record.get("target_lang") == config["target_lang"]]


def value_to_text(value: Any) -> str:
    return "" if value is None else str(value)


def text_to_nullable(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped.lower() == "null":
        return None
    return stripped


def metadata_items(record: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("qid", record.get("qid")),
        ("name", record.get("name")),
        ("target_lang", record.get("target_lang")),
        ("en_title", record.get("en_title")),
        ("target_title", record.get("target_title")),
        ("country_group", record.get("country_group")),
        ("occupations", ", ".join(record.get("occupations", [])) or None),
        ("sent_idx", record.get("sent_idx")),
    ]


def highlight_source_sentence(excerpt: str, source_sentence: str) -> str:
    excerpt_html = html.escape(excerpt or "")
    source_html = html.escape(source_sentence or "")
    if source_html and source_html in excerpt_html:
        highlighted = excerpt_html.replace(
            source_html,
            f"<mark class='source-highlight'>{source_html}</mark>",
            1,
        )
    else:
        highlighted = excerpt_html
    highlighted = highlighted.replace("\n", "<br>")
    return f"<div class='excerpt-box'>{highlighted}</div>"


def annotation_key(sample_id: str, claim_idx: int, field: str) -> str:
    return f"claim::{sample_id}::{claim_idx}::{field}"


def edit_widget_key(sample_id: str, claim_idx: int, field: str) -> str:
    return f"claim-edit::{EDIT_WIDGET_VERSION}::{sample_id}::{claim_idx}::{field}"


def missing_key(sample_id: str, missing_id: int, field: str) -> str:
    return f"missing::{sample_id}::{missing_id}::{field}"


def missing_ids_key(sample_id: str) -> str:
    return f"missing-ids::{sample_id}"


def missing_next_id_key(sample_id: str) -> str:
    return f"missing-next-id::{sample_id}"


def missing_active_key(sample_id: str, missing_id: int) -> str:
    return f"missing-active::{sample_id}::{missing_id}"


def sample_note_key(sample_id: str) -> str:
    return f"sample-note::{sample_id}"


def default_field_text(field: str, value: Any) -> str:
    if field == "hedge":
        if value is None:
            return "No"
        if isinstance(value, str) and (not value.strip() or value.strip().lower() == "null"):
            return "No"
    return value_to_text(value)


def empty_missing_claim_value(field: str) -> str:
    if field == "hedge":
        return "No"
    return ""


def missing_claim_has_content(missing_claim: dict[str, Any]) -> bool:
    for field, value in missing_claim.items():
        if field == "hedge":
            if value is not None and str(value).strip() not in {"", "No"}:
                return True
            continue
        if value is not None:
            return True
    return False


def hydrate_sample_state(
    record: dict[str, Any],
    existing_annotation: dict[str, Any] | None,
    dataset_id: str,
) -> None:
    sample_id = session_sample_id_for(dataset_id, record)
    existing_annotation = existing_annotation or {}
    existing_claim_annotations = {
        item.get("claim_idx"): item for item in existing_annotation.get("claim_annotations", [])
    }

    st.session_state.setdefault(
        sample_note_key(sample_id),
        existing_annotation.get("sample_note", ""),
    )

    for claim_idx, claim in enumerate(record.get("claims", [])):
        saved_claim = existing_claim_annotations.get(claim_idx, {})
        label_value = saved_claim.get("label") or ANNOTATION_OPTIONS[0]
        st.session_state.setdefault(annotation_key(sample_id, claim_idx, "label"), label_value)

        edited_claim = saved_claim.get("edited_claim") or {}
        if edited_claim:
            for field in EDITABLE_CLAIM_FIELDS:
                default_value = edited_claim.get(field, claim.get(field))
                st.session_state.setdefault(
                    edit_widget_key(sample_id, claim_idx, field),
                    default_field_text(field, default_value),
                )

    existing_missing_claims = existing_annotation.get("missing_claims", [])
    st.session_state.setdefault(
        missing_ids_key(sample_id),
        list(range(len(existing_missing_claims))),
    )
    st.session_state.setdefault(
        missing_next_id_key(sample_id),
        len(existing_missing_claims),
    )

    for missing_id, missing_claim in enumerate(existing_missing_claims):
        st.session_state.setdefault(missing_active_key(sample_id, missing_id), True)
        for field in EDITABLE_CLAIM_FIELDS:
            st.session_state.setdefault(
                missing_key(sample_id, missing_id, field),
                default_field_text(field, missing_claim.get(field)),
            )


def collect_current_sample_annotation(record: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    sample_id = session_sample_id_for(dataset_id, record)
    claim_annotations = []

    for claim_idx, claim in enumerate(record.get("claims", [])):
        label = st.session_state.get(annotation_key(sample_id, claim_idx, "label"), ANNOTATION_OPTIONS[0])

        claim_annotation = {
            "claim_idx": claim_idx,
            "label": None if label == ANNOTATION_OPTIONS[0] else label,
            "original_claim": deepcopy(claim),
            "edited_claim": None,
        }

        if claim_annotation["label"] == "Almost":
            edited_claim = {
                field: text_to_nullable(
                    st.session_state.get(edit_widget_key(sample_id, claim_idx, field), "")
                )
                for field in EDITABLE_CLAIM_FIELDS
            }
            edited_claim["source_sent"] = claim.get("source_sent") or record.get("source_sent")
            claim_annotation["edited_claim"] = edited_claim

        claim_annotations.append(claim_annotation)

    missing_claims = []
    for missing_id in st.session_state.get(missing_ids_key(sample_id), []):
        if not st.session_state.get(missing_active_key(sample_id, missing_id), False):
            continue
        missing_claim = {
            field: text_to_nullable(st.session_state.get(missing_key(sample_id, missing_id, field), ""))
            for field in EDITABLE_CLAIM_FIELDS
        }
        if missing_claim_has_content(missing_claim):
            missing_claim["source_sent"] = record.get("source_sent")
            missing_claims.append(missing_claim)

    annotation = {
        "sample_id": sample_id_for(record),
        "qid": record.get("qid"),
        "sent_idx": record.get("sent_idx"),
        "name": record.get("name"),
        "target_lang": record.get("target_lang"),
        "en_title": record.get("en_title"),
        "target_title": record.get("target_title"),
        "source_sent": record.get("source_sent"),
        "sample_note": text_to_nullable(st.session_state.get(sample_note_key(sample_id), "")),
        "claim_annotations": claim_annotations,
        "missing_claims": missing_claims,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return annotation


def ensure_active_sample_timer(language_code: str, annotator_id: str, record: dict[str, Any]) -> None:
    sample_key = active_time_sample_key(language_code, annotator_id)
    started_key = active_time_started_at_key(language_code, annotator_id)
    current_sample_id = sample_id_for(record)
    if st.session_state.get(sample_key) != current_sample_id:
        st.session_state[sample_key] = current_sample_id
        st.session_state[started_key] = now_ts()


def record_time_for_current_sample(
    meta_store: dict[str, Any],
    record: dict[str, Any],
    language_code: str,
    annotator_id: str,
    reset_start: bool = True,
) -> dict[str, Any]:
    sample_key = active_time_sample_key(language_code, annotator_id)
    started_key = active_time_started_at_key(language_code, annotator_id)
    current_sample_id = sample_id_for(record)
    started_at = st.session_state.get(started_key)

    if st.session_state.get(sample_key) != current_sample_id or started_at is None:
        st.session_state[sample_key] = current_sample_id
        st.session_state[started_key] = now_ts()
        return meta_store

    elapsed_seconds = max(0.0, now_ts() - float(started_at))
    time_by_sample = meta_store.setdefault("time_by_sample", {})
    sample_stats = time_by_sample.setdefault(
        current_sample_id,
        {
            "qid": record.get("qid"),
            "sent_idx": record.get("sent_idx"),
            "name": record.get("name"),
            "seconds": 0.0,
            "save_events": 0,
            "last_recorded_at": None,
        },
    )
    sample_stats["seconds"] = round(float(sample_stats.get("seconds", 0.0)) + elapsed_seconds, 2)
    sample_stats["save_events"] = int(sample_stats.get("save_events", 0)) + 1
    sample_stats["last_recorded_at"] = now_iso()
    meta_store["last_opened_at"] = now_iso()

    if reset_start:
        st.session_state[started_key] = now_ts()
        st.session_state[sample_key] = current_sample_id

    return meta_store


def annotation_has_content(annotation: dict[str, Any]) -> bool:
    if annotation.get("sample_note"):
        return True
    if annotation.get("missing_claims"):
        return True
    for item in annotation.get("claim_annotations", []):
        if item.get("label"):
            return True
        if item.get("edited_claim"):
            return True
    return False


def save_annotation_for_record(
    record: dict[str, Any],
    annotation_store: dict[str, dict[str, Any]],
    json_key: str,
    jsonl_key: str,
    dataset_id: str,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    annotation = collect_current_sample_annotation(record, dataset_id)
    sample_id = annotation["sample_id"]

    if annotation_has_content(annotation):
        annotation_store[sample_id] = annotation
        write_annotation_files(annotation_store, json_key, jsonl_key)
        return True, annotation_store

    if sample_id in annotation_store:
        del annotation_store[sample_id]
        write_annotation_files(annotation_store, json_key, jsonl_key)
        return True, annotation_store

    return False, annotation_store


def render_guidelines() -> None:
    with st.expander("Guidelines (selection + disambiguation + decomposition)", expanded=False):
        st.markdown(
            """
            **1. First decide whether the sentence expresses a verifiable proposition**

            - Judge only whether it is specific and verifiable, not whether it is true, false, or important.
            - Subjective praise, broad summaries, rhetoric, and speculation usually do not count as claims.
            - If a sentence mixes evaluative content with factual content, keep only the factual part.
            - Use the visible context to check whether the sentence is just an introduction or conclusion.

            **2. Claims should be as decontextualized as possible**

            - Resolve pronouns, partial names, abbreviations, and relative time/place expressions whenever the visible context supports a clear resolution.
            - Use only the biography subject, the visible excerpt, and the source sentence. Do not use outside knowledge.
            - If the context does not support a stable resolution, do not guess.
            - `source_sent` is the evidence sentence and should not be edited. Edit the extracted claim and its fields instead.

            **3. Decompose to the smallest useful fact-checking units**

            - One sentence may contain multiple claims, and they should be split when needed.
            - Do not split claims into trivial fragments that are not independently worth checking.
            - Each claim should represent one clear, self-contained factual unit.
            - `subject / predicate / object` are the core frame; `time / location / reason / manner / hedge` are supporting fields.

            **4. Label definitions**

            - `True`: the extracted claim is supported by the source sentence and is already well formed.
            - `Not a claim`: the extracted text should not have been treated as a claim in the first place.
            - `Almost`: the core idea is close, but the claim needs revision because it adds information, misses information, is not fully decontextualized, or uses imperfect field values.

            **5. What to do when you choose `Almost`**

            - Edit `claim / subject / predicate / object / time / location / reason / manner / hedge` in the revision box.
            - Use `null` or leave the field empty if the source sentence does not explicitly support that field.
            - If the claim text is close but the structured fields are wrong, choose `Almost` and correct the fields.

            **6. Infobox key explanations**

            - `claim`: the final standalone claim sentence that should be fact-checked.
            - `subject`: the main entity the claim is about.
            - `predicate`: the main relation, action, or state.
            - `object`: the main target, complement, or content of the predicate.
            - `time`: an explicit or clearly recoverable time expression.
            - `location`: an explicit or clearly recoverable place expression.
            - `reason`: a stated reason or purpose only when the sentence explicitly gives one.
            - `manner`: a stated means, method, or manner only when explicitly supported.
            - `hedge`: the exact uncertainty or attribution phrase, such as `reportedly` or `according to some accounts`; otherwise use `No`.

            **7. Missing claims**

            - If the source sentence contains a claim that was missed, add it in the `Missing Claims` section.
            - Fill any fields you can justify from the source sentence; leave unsupported fields as `null` or blank.
            - Missing claims should also be specific, verifiable, decontextualized when possible, and not over-decomposed.
            """
        )


def render_metadata(record: dict[str, Any]) -> None:
    st.subheader("Metadata")
    columns = st.columns(4)
    items = metadata_items(record)
    for index, (label, value) in enumerate(items):
        with columns[index % 4]:
            st.markdown(
                f"""
                <div style="padding:0.45rem 0.1rem 0.7rem 0.1rem;">
                  <div style="font-size:0.78rem; color:#5b6472; text-transform:uppercase; letter-spacing:0.04em;">{html.escape(str(label))}</div>
                  <div style="font-size:0.98rem; color:#17212b; line-height:1.45; word-break:break-word;">{html.escape(str(value if value is not None else "null"))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_claim_card(
    record: dict[str, Any],
    claim: dict[str, Any],
    claim_idx: int,
    dataset_id: str,
    saved_claim_annotation: dict[str, Any] | None,
    display_rank: int,
) -> None:
    sample_id = session_sample_id_for(dataset_id, record)
    card_title = f"Claim {display_rank}"
    with st.container(border=True):
        st.markdown(f"**{card_title}**")
        st.text_area(
            "Claim",
            value=claim.get("claim", ""),
            height=90,
            disabled=True,
            key=f"display-claim::{sample_id}::{claim_idx}",
        )

        st.radio(
            "Label",
            options=ANNOTATION_OPTIONS,
            key=annotation_key(sample_id, claim_idx, "label"),
            horizontal=True,
        )

        label = st.session_state.get(annotation_key(sample_id, claim_idx, "label"), ANNOTATION_OPTIONS[0])
        if label == "Almost":
            edited_claim = (saved_claim_annotation or {}).get("edited_claim") or {}
            for field in EDITABLE_CLAIM_FIELDS:
                widget_key = edit_widget_key(sample_id, claim_idx, field)
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = default_field_text(
                        field,
                        edited_claim.get(field, claim.get(field)),
                    )

            st.markdown("**Revised Infobox**")
            edited_columns = st.columns(2)
            for index, field in enumerate(EDITABLE_CLAIM_FIELDS):
                widget_key = edit_widget_key(sample_id, claim_idx, field)
                if field == "claim":
                    edited_columns[index % 2].text_area(
                        field,
                        key=widget_key,
                        height=100,
                        placeholder="Edit the revised claim text",
                    )
                else:
                    edited_columns[index % 2].text_input(
                        field,
                        key=widget_key,
                        placeholder="Default is No; edit only if the claim is hedged"
                        if field == "hedge"
                        else "Leave blank or use null if unsupported / not applicable",
                    )


def render_missing_claims(record: dict[str, Any], dataset_id: str) -> None:
    sample_id = session_sample_id_for(dataset_id, record)
    ids_key = missing_ids_key(sample_id)
    next_id_key = missing_next_id_key(sample_id)

    st.subheader("Missing Claims")
    if st.button("Add one", key=f"add-missing::{sample_id}"):
        missing_ids = list(st.session_state.get(ids_key, []))
        new_missing_id = st.session_state.get(next_id_key, 0)
        missing_ids.append(new_missing_id)
        st.session_state[ids_key] = missing_ids
        st.session_state[next_id_key] = new_missing_id + 1
        st.session_state[missing_active_key(sample_id, new_missing_id)] = False
        for field in EDITABLE_CLAIM_FIELDS:
            st.session_state[missing_key(sample_id, new_missing_id, field)] = empty_missing_claim_value(field)
        st.rerun()

    missing_ids = st.session_state.get(ids_key, [])
    if not missing_ids:
        st.caption("No missing claims added yet.")

    for display_idx, missing_id in enumerate(missing_ids, start=1):
        with st.container(border=True):
            with st.form(key=f"missing-form::{sample_id}::{missing_id}", clear_on_submit=False):
                is_active = st.session_state.get(missing_active_key(sample_id, missing_id), False)
                header_cols = st.columns([4.2, 1.4, 1.1, 1.3])
                header_cols[0].markdown(f"**Missing Claim {display_idx}**")
                status_class = "added" if is_active else "draft"
                status_label = "Added" if is_active else "Draft"
                header_cols[1].markdown(
                    f"<span class='status-badge {status_class}'>{status_label}</span>",
                    unsafe_allow_html=True,
                )
                button_label = "Update" if is_active else "Add"
                add_or_update_clicked = header_cols[2].form_submit_button(
                    button_label,
                    use_container_width=True,
                )
                remove_clicked = header_cols[3].form_submit_button(
                    "Remove",
                    use_container_width=True,
                )

                cols = st.columns(2)
                for index, field in enumerate(EDITABLE_CLAIM_FIELDS):
                    widget_key = missing_key(sample_id, missing_id, field)
                    if field == "claim":
                        cols[index % 2].text_area(
                            field,
                            key=widget_key,
                            height=100,
                            placeholder="Enter a missed claim",
                        )
                    else:
                        cols[index % 2].text_input(
                            field,
                            key=widget_key,
                            placeholder="Default is No; edit only if the claim is hedged"
                            if field == "hedge"
                            else "Default is null / leave blank if unsupported",
                        )

            if remove_clicked:
                st.session_state[ids_key] = [
                    current_id for current_id in st.session_state.get(ids_key, []) if current_id != missing_id
                ]
                st.session_state.pop(missing_active_key(sample_id, missing_id), None)
                for field in EDITABLE_CLAIM_FIELDS:
                    st.session_state.pop(missing_key(sample_id, missing_id, field), None)
                st.rerun()

            if add_or_update_clicked:
                st.session_state[missing_active_key(sample_id, missing_id)] = True
                st.rerun()


def main() -> None:
    st.markdown(
        """
        <style>
        .excerpt-box {
            padding: 1rem 1.1rem;
            border: 1px solid #d7dee7;
            border-radius: 0.8rem;
            background: #f8fafc;
            line-height: 1.7;
            font-size: 1rem;
        }
        .source-highlight {
            background: #ffe08a;
            padding: 0.1rem 0.2rem;
            border-radius: 0.25rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.34rem 0.78rem;
            border-radius: 999px;
            font-size: 0.98rem;
            font-weight: 700;
            line-height: 1;
            white-space: nowrap;
        }
        .status-badge.draft {
            background: #eef2f7;
            color: #4b5563;
        }
        .status-badge.added {
            background: #e7f8ee;
            color: #0f7a42;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Claim Extraction Annotation")
    st.caption("Use the left panel for context and the right panel to label extracted claims and add missing ones.")

    jsonl_files = discover_jsonl_files(Path.cwd())
    if not jsonl_files:
        st.error("No usable JSONL files were found in the current directory.")
        st.stop()

    resolved_source_files = resolve_source_files(jsonl_files)
    if "en" not in resolved_source_files or "target" not in resolved_source_files:
        st.error("Could not resolve both EN and TARGET source files from the current data directory.")
        st.write("Detected files:")
        for path in jsonl_files:
            st.write(f"- {path.relative_to(Path.cwd())}")
        st.stop()

    annotator_language = st.session_state.get(annotator_selection_key())
    annotator_id = st.session_state.get(annotator_id_key())
    existing_sessions = existing_registered_sessions()

    if annotator_language not in ANNOTATOR_CONFIGS or not annotator_id:
        st.subheader("Start or Resume Annotation")
        entry_mode = st.radio(
            "Mode",
            options=["First time here", "Resume annotation"],
            horizontal=True,
        )
        input_annotator_id = st.text_input(
            "Annotator ID",
            placeholder="Letters and numbers only, 3-20 characters",
        ).strip()
        selected_language = st.radio(
            "Annotator language",
            options=list(ANNOTATOR_CONFIGS.keys()),
            horizontal=False,
        )
        st.caption("Please remember your Annotator ID. You will need it to resume after closing the app.")

        if st.button("Continue", type="primary"):
            validation_error = validate_annotator_id(input_annotator_id)
            if validation_error:
                st.error(validation_error)
                st.stop()

            selected_language_code = language_code_for_annotator(selected_language)
            selected_data_path = resolved_source_files[ANNOTATOR_CONFIGS[selected_language]["route"]]
            selected_meta_s3_key = user_meta_s3_key(selected_language_code, input_annotator_id)

            if entry_mode == "First time here":
                if input_annotator_id in existing_sessions:
                    st.error("This Annotator ID already exists. Please choose a different one or use Resume annotation.")
                    st.stop()

                meta_store = build_initial_meta_store(
                    input_annotator_id,
                    selected_language,
                    selected_language_code,
                    selected_data_path,
                )
                write_meta_file(meta_store, selected_meta_s3_key)
            else:
                if not s3_object_exists(selected_meta_s3_key):
                    st.error("No existing annotation session was found for this Annotator ID and language.")
                    st.stop()

                meta_store = load_meta_file(selected_meta_s3_key)
                if meta_store.get("annotator_language") != selected_language:
                    st.error("This Annotator ID exists, but not for the selected language.")
                    st.stop()
                meta_store["last_opened_at"] = now_iso()
                write_meta_file(meta_store, selected_meta_s3_key)

            st.session_state[annotator_selection_key()] = selected_language
            st.session_state[annotator_id_key()] = input_annotator_id
            st.session_state[annotator_mode_key()] = entry_mode
            st.rerun()
        st.stop()

    annotator_config = ANNOTATOR_CONFIGS[annotator_language]
    language_code = language_code_for_annotator(annotator_language)
    data_path = resolved_source_files[annotator_config["route"]]
    all_records = load_jsonl_records(str(data_path))
    records = filtered_records_for_annotator(all_records, annotator_language)
    if not records:
        st.error(f"No records matched the annotator language route for {annotator_language}.")
        st.stop()
    dataset_id = data_path.stem
    annotation_s3_json_key, annotation_s3_jsonl_key = user_annotation_s3_keys(language_code, annotator_id)
    meta_s3_key = user_meta_s3_key(language_code, annotator_id)
    session_signature = f"{annotator_id}::{language_code}"

    if st.session_state.get("_active_session_signature") != session_signature:
        st.session_state["_active_session_signature"] = session_signature
        st.session_state["_annotation_store"] = load_saved_annotations(annotation_s3_json_key)
        st.session_state["_meta_store"] = load_meta_file(meta_s3_key)

    annotation_store = st.session_state["_annotation_store"]
    meta_store = st.session_state.get("_meta_store", {})
    saved_ids = saved_sample_ids(records, annotation_store)
    index_key = f"current-index::{data_path.stem}::{annotator_language}::{annotator_id}"
    st.session_state.setdefault(index_key, 0)
    st.session_state[index_key] = min(st.session_state[index_key], max(len(records) - 1, 0))
    current_idx = st.session_state[index_key]
    current_record = records[current_idx]
    current_sample_id = sample_id_for(current_record)
    current_session_sample_id = session_sample_id_for(dataset_id, current_record)
    current_saved_annotation = annotation_store.get(current_sample_id)
    hydrate_sample_state(current_record, current_saved_annotation, dataset_id)
    ensure_active_sample_timer(language_code, annotator_id, current_record)
    current_is_saved = current_sample_id in saved_ids

    st.sidebar.markdown("### Data and Saving")
    st.sidebar.write(f"Annotator ID: `{annotator_id}`")
    st.sidebar.write(f"Annotator language: `{annotator_language}`")
    st.sidebar.write(f"Source file: `{describe_data_file(data_path)}`")
    if annotator_config["route"] == "target":
        st.sidebar.write(
            f"Filtered view: `target_lang={annotator_config['target_lang']}` | `country_group={annotator_config['country_group']}`"
        )
    else:
        st.sidebar.write("Filtered view: `All 300 EN biography samples`")
    st.sidebar.write(f"Samples: `{len(records)}`")
    st.sidebar.write(f"Saved: `{len(saved_ids)}`")
    st.sidebar.write(f"S3 key: `{annotation_s3_json_key}`")
    st.sidebar.write(f"S3 meta: `{meta_s3_key}`")
    unsaved_index = next_unsaved_index(records, annotation_store)
    if unsaved_index is None:
        st.sidebar.write("Next unsaved: `All saved`")
    else:
        st.sidebar.write(f"Next unsaved: `{unsaved_index + 1}`")

    if st.sidebar.button("Jump to next unsaved", use_container_width=True, disabled=unsaved_index is None):
        st.session_state[index_key] = unsaved_index
        st.rerun()

    if st.sidebar.button("Switch annotator / language", use_container_width=True):
        st.session_state.pop(annotator_selection_key(), None)
        st.session_state.pop(annotator_id_key(), None)
        st.session_state.pop(annotator_mode_key(), None)
        st.rerun()

    jump_value = st.sidebar.number_input(
        "Jump to sample (1-based)",
        min_value=1,
        max_value=max(1, len(records)),
        value=current_idx + 1,
        step=1,
    )
    if st.sidebar.button("Jump", use_container_width=True):
        st.session_state[index_key] = int(jump_value) - 1
        st.rerun()

    if st.sidebar.button("Save current sample", use_container_width=True):
        meta_store = record_time_for_current_sample(meta_store, current_record, language_code, annotator_id)
        changed, annotation_store = save_annotation_for_record(
            current_record,
            annotation_store,
            annotation_s3_json_key,
            annotation_s3_jsonl_key,
            dataset_id,
        )
        st.session_state["_annotation_store"] = annotation_store
        if changed:
            meta_store["last_saved_at"] = now_iso()
        meta_store["last_opened_at"] = now_iso()
        st.session_state["_meta_store"] = meta_store
        write_meta_file(meta_store, meta_s3_key)
        if changed:
            saved_ids = saved_sample_ids(records, annotation_store)
            current_is_saved = current_sample_id in saved_ids
            st.sidebar.success("Saved current sample")
        else:
            st.sidebar.info("No annotation content yet, so nothing was written")

    if s3_object_exists(annotation_s3_json_key):
        annotation_json_content = json.dumps(
            s3_read_json(annotation_s3_json_key), ensure_ascii=False, indent=2, sort_keys=True
        )
        stem = f"claw_4_sent_annotations_{language_code}_by_{annotator_id}"
        st.sidebar.download_button(
            "Download annotations.json",
            data=annotation_json_content,
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown(f"### Sample {current_idx + 1} / {len(records)}")
    if current_is_saved:
        st.success("Saved annotation exists for this sample.")
    else:
        st.warning("This sample has not been manually saved yet.")
    render_guidelines()
    render_metadata(current_record)

    nav_cols = st.columns([1, 1, 6])
    if nav_cols[0].button("Previous", disabled=current_idx == 0, use_container_width=True):
        st.session_state[index_key] = max(0, current_idx - 1)
        st.rerun()
    if nav_cols[1].button("Next", disabled=current_idx >= len(records) - 1, use_container_width=True):
        st.session_state[index_key] = min(len(records) - 1, current_idx + 1)
        st.rerun()

    left_col, right_col = st.columns([1.05, 1.2], gap="large")

    with left_col:
        st.subheader("Excerpt")
        st.markdown(
            highlight_source_sentence(
                current_record.get("excerpt", ""),
                current_record.get("source_sent", ""),
            ),
            unsafe_allow_html=True,
        )
        st.subheader("Source Sentence")
        st.text_area(
            "source_sent",
            value=current_record.get("source_sent", ""),
            height=140,
            disabled=True,
        )
        st.subheader("Sample Notes")
        st.text_area(
            "sample_note",
            key=sample_note_key(current_session_sample_id),
            height=100,
            placeholder="Example: this sentence needs a different decomposition, context is insufficient, or source sentence does not exactly match the excerpt.",
        )

    with right_col:
        st.subheader("Claims")
        claims = current_record.get("claims", [])
        if not claims:
            st.info("This sample has no model-generated claims.")
        saved_claim_annotations = {
            item.get("claim_idx"): item
            for item in (current_saved_annotation or {}).get("claim_annotations", [])
        }
        sorted_claim_items = sorted(
            list(enumerate(claims)),
            key=lambda item: (-(len((item[1] or {}).get("claim", "") or "")), item[0]),
        )
        for display_rank, (claim_idx, claim) in enumerate(sorted_claim_items, start=1):
            render_claim_card(
                current_record,
                claim,
                claim_idx,
                dataset_id,
                saved_claim_annotations.get(claim_idx),
                display_rank,
            )

        render_missing_claims(current_record, dataset_id)


if __name__ == "__main__":
    main()