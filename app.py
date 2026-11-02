"""
ns-nano backend: Neurosynth functional-dissociation API (Flask + PostgreSQL)
Psychoinformatics & Neuroinformatics, Week 8 sprint.

Endpoints
---------
GET /terms/<term>/studies                      studies whose abstract uses <term>
GET /locations/<x_y_z>/studies                 studies reporting a peak within r mm of [x, y, z]
GET /dissociate/terms/<term_a>/<term_b>        studies that use term_a but NOT term_b
GET /dissociate/locations/<x1_y1_z1>/<x2_y2_z2> studies with a peak near A but none near B

Optional query parameters
-------------------------
threshold  TF-IDF cut-off for "a study mentions a term" (default 0.001, as in Neurosynth)
r          search radius in mm for coordinates (default 6, as in Neurosynth's location pages)
limit      maximum number of studies to list (default: all; counts always cover all studies)

Helpers: /  (HTML index with clickable examples) · /terms?q=<text> (vocabulary search)
         /test_db (database smoke test) · /img (Tren's smoke-test image, if present)

Configuration: the DB_URL environment variable (DATABASE_URL is also accepted).
"""
import math
import os
import re
from functools import lru_cache
from urllib.parse import unquote

from flask import Flask, jsonify, request, send_file
from sqlalchemy import create_engine, text

SCHEMA = "ns"
DEFAULT_THRESHOLD = 0.001   # TF-IDF >= 0.001 counts as "term present" (Neurosynth convention)
DEFAULT_RADIUS = 6.0        # mm
MAX_RADIUS = 30.0

# --------------------------------------------------------------------------- database
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("Missing DB_URL environment variable.")
        if url.startswith("postgres://"):                      # old Heroku/Render style scheme
            url = "postgresql://" + url[len("postgres://"):]
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=2)
    return _engine


def query(sql, **params):
    """Run one SQL statement and return a list of dicts."""
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


# --------------------------------------------------------------------------- input parsing
class BadRequest(Exception):
    def __init__(self, message, status=400, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


def parse_term(raw):
    """URL path segment -> vocabulary term: 'posterior_cingulate' / 'Posterior%20Cingulate' -> 'posterior cingulate'."""
    term = re.sub(r"\s+", " ", unquote(raw).replace("_", " ")).strip().lower()
    if not term:
        raise BadRequest("Empty term.")
    return term


def parse_xyz(raw):
    """'0_-52_26' -> (0.0, -52.0, 26.0)."""
    parts = unquote(raw).replace(",", "_").split("_")
    try:
        xyz = tuple(float(p) for p in parts)
    except ValueError:
        xyz = ()
    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
        raise BadRequest(f"Coordinates must look like x_y_z, e.g. 0_-52_26 (got '{raw}').")
    return xyz


def arg_float(name, default, lo, hi):
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise BadRequest(f"Query parameter '{name}' must be a number.")
    if not lo <= v <= hi:
        raise BadRequest(f"Query parameter '{name}' must be between {lo} and {hi}.")
    return v


def arg_limit():
    raw = request.args.get("limit")
    if raw in (None, "", "all", "0"):
        return None
    if not raw.isdigit():
        raise BadRequest("Query parameter 'limit' must be a positive integer.")
    return int(raw)


@lru_cache(maxsize=4096)
def term_exists(term):
    return bool(query(f"SELECT 1 FROM {SCHEMA}.terms WHERE term = :t", t=term))


def similar_terms(term, n=10):
    words = [w for w in term.split() if len(w) > 2] or [term]
    pattern = "%" + "%".join(words) + "%"
    rows = query(f"""SELECT term FROM {SCHEMA}.terms WHERE term ILIKE :p
                     ORDER BY n_studies DESC LIMIT :n""", p=pattern, n=n)
    if not rows:   # fall back to any single word
        rows = query(f"""SELECT term FROM {SCHEMA}.terms WHERE term ILIKE ANY(:ps)
                         ORDER BY n_studies DESC LIMIT :n""", ps=[f"%{w}%" for w in words], n=n)
    return [r["term"] for r in rows]


def require_term(term):
    if not term_exists(term):
        raise BadRequest(f"Term '{term}' is not in the Neurosynth vocabulary.", status=404,
                         suggestions=similar_terms(term))


# --------------------------------------------------------------------------- SQL building blocks
# A study "mentions" a term when its abstract TF-IDF weight >= threshold.
TERM_SET = f"SELECT study_id, weight FROM {SCHEMA}.annotations_terms WHERE term = :{{p}}term AND weight >= :thr"

# A study "mentions" a location when it reports >= 1 peak within r mm.
# The BETWEEN box uses the B-tree indexes on x / y / z; the exact sphere test runs on the few rows left.
NEAR_SET = f"""
    SELECT study_id, MIN(SQRT(POWER(x - :{{p}}x, 2) + POWER(y - :{{p}}y, 2) + POWER(z - :{{p}}z, 2))) AS distance
    FROM {SCHEMA}.coordinates
    WHERE x BETWEEN :{{p}}x - :r AND :{{p}}x + :r
      AND y BETWEEN :{{p}}y - :r AND :{{p}}y + :r
      AND z BETWEEN :{{p}}z - :r AND :{{p}}z + :r
      AND POWER(x - :{{p}}x, 2) + POWER(y - :{{p}}y, 2) + POWER(z - :{{p}}z, 2) <= POWER(:r, 2)
    GROUP BY study_id"""

META_COLS = "m.study_id, m.title, m.authors, m.journal, m.year"


def term_set(prefix):
    return TERM_SET.format(p=prefix)


def near_set(prefix):
    return NEAR_SET.format(p=prefix)


def set_counts(set_a, set_b, params):
    """|A|, |B|, |A ∩ B|, |A \\ B|, |B \\ A| in one round trip (set logic done by PostgreSQL)."""
    row = query(f"""
        WITH a AS ({set_a}), b AS ({set_b})
        SELECT (SELECT COUNT(*) FROM a) AS a,
               (SELECT COUNT(*) FROM b) AS b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM a INTERSECT SELECT study_id FROM b) s) AS a_and_b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM a EXCEPT    SELECT study_id FROM b) s) AS a_not_b,
               (SELECT COUNT(*) FROM (SELECT study_id FROM b EXCEPT    SELECT study_id FROM a) s) AS b_not_a
    """, **params)[0]
    return {k: int(v) for k, v in row.items()}


def clean(rows):
    for r in rows:
        for k in ("weight", "distance"):
            if r.get(k) is not None:
                r[k] = round(float(r[k]), 4 if k == "weight" else 2)
        if r.get("year") is not None:
            r["year"] = int(r["year"])
    return rows


# --------------------------------------------------------------------------- app
def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.errorhandler(BadRequest)
    def bad_request(e):
        return jsonify({"error": str(e), **e.extra}), e.status

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Unknown endpoint. See / for the list of endpoints."}), 404

    @app.errorhandler(Exception)
    def server_error(e):
        app.logger.exception(e)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

    # ---------------------------------------------------------------- 1) studies by term
    @app.get("/terms/<path:term>/studies")
    def studies_by_term(term):
        term = parse_term(term)
        require_term(term)
        thr = arg_float("threshold", DEFAULT_THRESHOLD, 0.0, 1.0)
        rows = query(f"""
            SELECT {META_COLS}, a.weight
            FROM ({term_set('a')}) a JOIN {SCHEMA}.metadata m USING (study_id)
            ORDER BY a.weight DESC, m.study_id
            LIMIT :lim""", aterm=term, thr=thr, lim=arg_limit())
        n = query(f"SELECT COUNT(*) AS n FROM ({term_set('a')}) a", aterm=term, thr=thr)[0]["n"]
        return jsonify({"query": {"term": term, "threshold": thr},
                        "count": int(n), "studies": clean(rows)})

    # ---------------------------------------------------------------- 2) studies by location
    @app.get("/locations/<coords>/studies")
    def studies_by_location(coords):
        x, y, z = parse_xyz(coords)
        r = arg_float("r", DEFAULT_RADIUS, 0.0, MAX_RADIUS)
        p = dict(ax=x, ay=y, az=z, r=r)
        rows = query(f"""
            SELECT {META_COLS}, a.distance
            FROM ({near_set('a')}) a JOIN {SCHEMA}.metadata m USING (study_id)
            ORDER BY a.distance, m.study_id
            LIMIT :lim""", **p, lim=arg_limit())
        n = query(f"SELECT COUNT(*) AS n FROM ({near_set('a')}) a", **p)[0]["n"]
        return jsonify({"query": {"location": [x, y, z], "radius_mm": r},
                        "count": int(n), "studies": clean(rows)})

    # ---------------------------------------------------------------- 3) dissociate by terms
    @app.get("/dissociate/terms/<path:term_a>/<path:term_b>")
    def dissociate_terms(term_a, term_b):
        a, b = parse_term(term_a), parse_term(term_b)
        for t in (a, b):
            require_term(t)
        thr = arg_float("threshold", DEFAULT_THRESHOLD, 0.0, 1.0)
        p = dict(aterm=a, bterm=b, thr=thr)
        rows = query(f"""
            WITH a AS ({term_set('a')}), b AS ({term_set('b')}),
                 d AS (SELECT study_id FROM a EXCEPT SELECT study_id FROM b)
            SELECT {META_COLS}, a.weight
            FROM d JOIN a USING (study_id) JOIN {SCHEMA}.metadata m USING (study_id)
            ORDER BY a.weight DESC, m.study_id
            LIMIT :lim""", **p, lim=arg_limit())
        counts = set_counts(term_set("a"), term_set("b"), p)
        return jsonify({"query": {"term_a": a, "term_b": b, "threshold": thr},
                        "description": f"Studies that mention '{a}' but not '{b}'",
                        "counts": counts, "count": counts["a_not_b"], "studies": clean(rows)})

    # ---------------------------------------------------------------- 4) dissociate by locations
    @app.get("/dissociate/locations/<coords_a>/<coords_b>")
    def dissociate_locations(coords_a, coords_b):
        (ax, ay, az), (bx, by, bz) = parse_xyz(coords_a), parse_xyz(coords_b)
        r = arg_float("r", DEFAULT_RADIUS, 0.0, MAX_RADIUS)
        p = dict(ax=ax, ay=ay, az=az, bx=bx, by=by, bz=bz, r=r)
        counts = set_counts(near_set("a"), near_set("b"), p)
        rows = query(f"""
            WITH a AS ({near_set('a')}), b AS ({near_set('b')}),
                 d AS (SELECT study_id FROM a EXCEPT SELECT study_id FROM b)
            SELECT {META_COLS}, a.distance
            FROM d JOIN a USING (study_id) JOIN {SCHEMA}.metadata m USING (study_id)
            ORDER BY a.distance, m.study_id
            LIMIT :lim""", **p, lim=arg_limit())
        return jsonify({"query": {"location_a": [ax, ay, az], "location_b": [bx, by, bz], "radius_mm": r},
                        "description": f"Studies reporting a peak within {r:g} mm of {[ax, ay, az]} "
                                       f"but none within {r:g} mm of {[bx, by, bz]}",
                        "counts": counts, "count": counts["a_not_b"], "studies": clean(rows)})

    # ---------------------------------------------------------------- helpers
    @app.get("/terms")
    def search_terms():
        q = request.args.get("q", "").strip().lower()
        rows = query(f"""SELECT term, n_studies FROM {SCHEMA}.terms
                         WHERE (:q = '' OR term ILIKE :p) ORDER BY n_studies DESC LIMIT :n""",
                     q=q, p=f"%{q}%", n=int(request.args.get("n", 50)))
        return jsonify({"query": q, "terms": rows})

    @app.get("/test_db")
    def test_db():
        out = {"ok": False}
        try:
            out["version"] = query("SELECT version() AS v")[0]["v"]
            for t in ("metadata", "coordinates", "annotations_terms", "terms"):
                out[f"{t}_count"] = int(query(f"SELECT COUNT(*) AS n FROM {SCHEMA}.{t}")[0]["n"])
            out["coordinates_sample"] = query(f"SELECT study_id, x, y, z FROM {SCHEMA}.coordinates LIMIT 3")
            out["ok"] = True
            return jsonify(out)
        except Exception as e:
            out["error"] = str(e)
            return jsonify(out), 500

    @app.get("/img")
    def show_img():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amygdala.gif")
        if not os.path.exists(path):
            return jsonify({"error": "amygdala.gif not deployed"}), 404
        return send_file(path, mimetype="image/gif")

    @app.get("/")
    def index():
        ex = [
            ("/terms/amygdala/studies?limit=20", "Studies about the amygdala"),
            ("/locations/-22_0_-20/studies?limit=20", "Studies with a peak near the left amygdala"),
            ("/dissociate/terms/posterior_cingulate/ventromedial_prefrontal", "posterior cingulate \\ ventromedial prefrontal"),
            ("/dissociate/terms/ventromedial_prefrontal/posterior_cingulate", "ventromedial prefrontal \\ posterior cingulate"),
            ("/dissociate/locations/0_-52_26/-2_50_-6", "PCC [0,-52,26] \\ vmPFC [-2,50,-6]"),
            ("/dissociate/locations/-2_50_-6/0_-52_26", "vmPFC [-2,50,-6] \\ PCC [0,-52,26]"),
            ("/terms?q=cingulate", "Vocabulary search"),
            ("/test_db", "Database smoke test"),
        ]
        items = "".join(f'<li><a href="{u}"><code>{u}</code></a> — {d}</li>' for u, d in ex)
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>ns-nano API</title>
<style>body{{font-family:system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 16px;line-height:1.5}}
code{{background:#f3f3f1;padding:1px 4px;border-radius:4px}}</style></head><body>
<h1>ns-nano: Neurosynth functional-dissociation API</h1>
<p>Flask + PostgreSQL backend over 14,371 Neurosynth studies, 507,891 peak coordinates and 3,228 abstract terms.</p>
<h3>Endpoints</h3><ul>
<li><code>/terms/&lt;term&gt;/studies</code></li><li><code>/locations/&lt;x_y_z&gt;/studies</code></li>
<li><code>/dissociate/terms/&lt;term_a&gt;/&lt;term_b&gt;</code> — studies mentioning term_a but not term_b</li>
<li><code>/dissociate/locations/&lt;x1_y1_z1&gt;/&lt;x2_y2_z2&gt;</code> — studies with a peak near A but none near B</li></ul>
<p>Options: <code>?threshold=0.001</code> (TF-IDF cut-off), <code>?r=6</code> (radius, mm), <code>?limit=50</code>.
Use underscores or %20 for spaces in terms.</p>
<h3>Examples</h3><ul>{items}</ul></body></html>"""

    return app


app = create_app()
