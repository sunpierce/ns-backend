# ns-nano backend: Neurosynth functional dissociation API

Flask + PostgreSQL backend for the Week 8 sprint. It builds on [spiketren/ns-nano](https://github.com/spiketren/ns-nano).

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /terms/<term>/studies` | Studies whose abstract uses `<term>` (TF-IDF ≥ threshold), sorted by weight |
| `GET /locations/<x_y_z>/studies` | Studies that report a peak within `r` mm of `[x, y, z]`, sorted by distance |
| `GET /dissociate/terms/<term_a>/<term_b>` | Studies that mention **term_a but not term_b** (A \ B) |
| `GET /dissociate/locations/<x1_y1_z1>/<x2_y2_z2>` | Studies with a peak near **A but none near B** |

**Query parameters (all optional):**
* `threshold` sets the TF-IDF cut-off for "mentions a term". The default is `0.001`, as in Neurosynth.
* `r` sets the search radius in mm. The default is `6`.
* `limit` caps how many studies are listed. Counts always cover all studies.

**Writing terms in the URL:** use `_` or `%20` for spaces. Case does not matter.

**What a dissociation response contains:**
* `counts`: |A|, |B|, |A∩B|, |A\B| and |B\A|, so the reverse direction is also available.
* `count`: the number of studies in A \ B.
* `studies`: id, title, authors, journal, year, and weight or distance.

**Helper endpoints:**
* `/`: an index page with clickable examples.
* `/terms?q=cingul`: vocabulary search.
* `/test_db`: a database smoke test.
* `/img`: Tren's smoke-test GIF.

**Default Mode Network test cases:**
```
/dissociate/terms/posterior_cingulate/ventromedial_prefrontal
/dissociate/terms/ventromedial_prefrontal/posterior_cingulate
/dissociate/locations/0_-52_26/-2_50_-6
/dissociate/locations/-2_50_-6/0_-52_26
```

## Design

* **Set logic runs inside PostgreSQL.** A \ B is a SQL `EXCEPT`, and the counts use `INTERSECT` and `EXCEPT` in one query.
* **Sparse term table.** `ns.annotations_terms(term, study_id, weight)` stores only non-zero TF-IDF values, with primary key `(term, study_id)`. Fetching a term's study set is therefore an index range scan.
* **Coordinates without PostGIS.** Coordinates stay as plain `x, y, z` columns in MNI millimetres, each with a B-tree index.
  1. A sphere query first applies a ±r box, which the indexes serve.
  2. It then runs the exact Euclidean test on the few rows left.
  This avoids storing MNI millimetres as if they were latitude and longitude (SRID 4326).
* **Vocabulary table and full-text search.** `ns.terms` validates terms; an unknown term returns 404 with suggestions. `ns.metadata.fts` is a generated `tsvector` column with a GIN index, ready for full-text search.

## Deploy on Render

1. **Push this folder to a GitHub repository.**
2. **Create the database and web service.** In Render, choose **New → Blueprint** and select the repository. `render.yaml` creates:
   * a free PostgreSQL database `ns-db`;
   * the web service, with `DB_URL` already wired to the database.

   To set things up by hand instead:
   * create a PostgreSQL database;
   * create a Web Service with build command `pip install -r requirements.txt` and start command `gunicorn app:app --bind 0.0.0.0:$PORT`;
   * set the environment variable `DB_URL` to the database's **Internal Database URL**.
3. **Load the data once, from your own computer.**
   1. Copy the database's **External Database URL** from Render.
   2. Download `coordinates.parquet`, `metadata.parquet` and `annotations.parquet` from Tren's repo into `./data`.
   3. Run:
   ```bash
   pip install -r requirements-load.txt
   python create_db.py --url "<External Database URL>" --data-dir ./data
   ```
4. **Check that it works.** Open `https://<service>.onrender.com/test_db`. It should report 14,371 studies and 507,891 coordinates. Then try the four test cases above.

Free Render services go to sleep after a period of inactivity, so the first request can take 30–60 seconds.
