# Daily Report — Feb 22, 2026

## Objective

Test and validate the full profile search pipeline — bypass the broken media-recognizer Go service and prove direct access to Milvus (vector search) + MongoDB (metadata) + S3 (profile images) works end-to-end.

---

## 1. media-recognizer Go Service — Still Broken

Restarted the `media-recognizer-aws-milvus` pod and re-tested the identify API.

**Error persists after pod restart:**

```
POST /generic/identify → 500 (60s timeout)
"failed get target partitions: failed to query first record.
 collName buffalo_l: rpc error: code = Canceled desc = context canceled"
```

**Root cause discovered:** The `buffalo_l` collection has **1,024 partitions** (`_default_0` through `_default_1023`). The Go Milvus SDK's partition discovery mechanism can't handle this many partitions — it tries to query a first record from each partition, times out after 60s.

| Test | Result |
|------|--------|
| media-recognizer Go API | 60s timeout, 500 error |
| Direct pymilvus (same collection) | 0.39s, works perfectly |

The Go SDK is the problem, not Milvus.

---

## 2. Direct Bypass — Full Pipeline Tested

Bypassed media-recognizer entirely. Tested direct access to all 3 backing services from the WARP container.

### End-to-End Test Results

| Step | Service | Time | Result |
|------|---------|------|--------|
| 1. Vector search | Milvus (pymilvus) | **0.4s** | 5 matches from 1M+ faces |
| 2. Metadata lookup | MongoDB (pymongo) | **1.7s** | 5/5 profiles found |
| 3. Image download | S3 (boto3) | **2.2s** | 5/5 images downloaded |
| **Total** | | **5.3s** | Full pipeline working |

### Sample Output

```
#1  76.93% | Pericles 'Perry' Abbasi | Twitter | @ElectionLegal
#2  75.93% | John M. Comeau          | Twitter | @23238365
#3  74.98% | Grandmaj (Judy)         | Twitter | @567911598
#4  73.98% | Mark Woodard            | Twitter | @59517726
#5  73.37% | Amy No                  | Twitter | @29376810
```

---

## 3. Deep Exploration — 15 Tests

### Embedding Validation

| Test | Result |
|------|--------|
| Dimensions | 512-D |
| L2 Norm | 1.0000 (perfectly normalized ArcFace) |
| Random vector test | Top match = 19.95% (confirms real matches at 77% are meaningful) |

### Similarity Score Distribution (top 50 results)

| Rank | Similarity |
|------|-----------|
| #1 | 76.9% |
| #5 | 73.4% |
| #10 | 69.5% |
| #25 | 68.3% |
| #50 | 65.4% |

**Recommended threshold: 70%** — random noise scores ~20%, true matches ~77%, clear separation.

### Performance

| Test | Result |
|------|--------|
| Single search latency | 385ms avg (10 runs, min 357ms, max 440ms) |
| Batch search (5 at once) | 400ms total = **80ms per query** |
| Latency consistency | Very stable, no outliers |

### Milvus ↔ MongoDB Coverage

| Test | Result |
|------|--------|
| 20 random Milvus entity_ids checked | **20/20 found in MongoDB** (100%) |
| MongoDB `_entity_id` index | B-tree + text index (fast lookups) |
| `center` field | All top results are `center=true` (representative face per person) |

### Platform Distribution (Full Count)

| Platform | Profiles | Fields Always Present |
|----------|----------|-----------------------|
| Twitter | **538,181** | name, id, profileImage, profileUrl |
| Instagram | **488,821** | name, id, profileImage (+ bio/followers for 18%) |
| TikTok | **236,491** | name, id, profileImage |
| Facebook | 0 | — |
| **Total** | **1,263,495** | |

### Cross-Platform Linking

| Feature | Status |
|---------|--------|
| `_group_id` | Only 5 entries (essentially unused) |
| `_parent_id` | 0 entries |
| Groups collection | 1 group, 4 members |

Each MongoDB entry = one person on one platform. No cross-platform linking exists.

### All Milvus Collections on AWS (172.31.30.225)

| Collection | Rows | Dims | Model | Status |
|------------|------|------|-------|--------|
| **buffalo_l** | **1,021,778** | 512 | ArcFace | Loaded (mmap), working |
| siglip2_base_patch16_512 | 1,185,418 | 768 | CLIP-like | Released |
| buffalo_sc | 0 | 512 | — | Empty |
| speechbrain_spkrec_ecapa | 0 | 192 | — | Empty (voice) |

### S3 Profile Images

| Detail | Value |
|--------|-------|
| Bucket | `milvus-storage-2025` |
| Region | `us-east-1` |
| Direct SDK download | Works (20-40KB per image) |
| Presigned URLs | Blocked by bucket policy (403) |
| Solution | Proxy through our API or serve as base64 |

---

## 4. MongoDB Schema — GenericFeatures Collection

```
{
  "_id": ObjectId,
  "id": String,
  "_entity_id": "089cac6fdc9b1267...",     ← matches Milvus _entity_id
  "_parent_id": null,
  "_recognizer_id": "siglip2_base_...",     ← recognizer that ingested it
  "_group_id": "htSAYAwdOIsAD9yz",          ← rare, cross-platform grouping
  "metadata": {
    "twitterData": {                         ← platform-specific key
      "name": "Pericles 'Perry' Abbasi",
      "id": "2359971372",
      "profileUrl": "https://twitter.com/ElectionLegal",
      "profileImage": "profile_images/089cac6f...jpg"
    }
  },
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Platform metadata keys:** `twitterData`, `tiktokData`, `instagramData`

**Instagram bonus fields (18% of entries):** `biography`, `follower_count`, `following_count`, `media_count`, `category`, `full_name`, `external_url`, `summary`

---

## 5. Connection Details for Implementation

Our `face-ocr-api` will need 3 clients (all via WARP VPN sidecar):

| Service | Client | Connection String |
|---------|--------|-------------------|
| Milvus | `pymilvus.MilvusClient` | `http://172.31.30.225:19530` |
| MongoDB | `pymongo.MongoClient` | `mongodb://serveradmin:<MONGO_PASSWORD>@<MILVUS_HOST>:27017` |
| S3 | `boto3.client('s3')` | `s3.us-east-1.amazonaws.com` (key: `<AWS_ACCESS_KEY_ID>`) |

**MongoDB database:** `media_recognizer`
**MongoDB collection:** `GenericFeatures` (1,263,495 docs)
**Milvus collection:** `buffalo_l` (1,021,778 vectors, 512-D, COSINE, IVF_FLAT)

---

## 6. Architecture Decision — Bypass media-recognizer

### Before (Broken)

```
face-ocr-api → media-recognizer (Go) → Milvus + MongoDB
                      ❌ Go SDK fails on 1024 partitions
```

### After (Direct Access)

```
face-ocr-api → pymilvus  → Milvus (0.4s)    ✅
             → pymongo   → MongoDB (1.7s)    ✅
             → boto3     → S3 images (2.2s)  ✅
```

### Why Bypass

1. media-recognizer Go SDK can't handle 1024 partitions (60s timeout)
2. Direct pymilvus works perfectly (0.4s)
3. We control the code — can debug, tune, optimize
4. No dependency on team fixing the Go service
5. Same data, same results — media-recognizer is just a wrapper around these DBs

---

## 7. Blockers Resolved Today

| Blocker | Status | Resolution |
|---------|--------|------------|
| media-recognizer Go SDK partition timeout | Bypassed | Direct pymilvus/pymongo |
| MongoDB metadata accessibility | Resolved | 100% entity coverage confirmed |
| S3 image accessibility | Resolved | boto3 SDK download works (presigned URLs blocked) |
| Embedding compatibility | Confirmed | buffalo_l = ArcFace 512-D COSINE, same as our pipeline |
| Milvus mmap (from previous session) | Stable | 385ms consistent latency |

---

## Next Steps

1. **Implement `ProfileSearchService`** in face-ocr-api — pymilvus + pymongo + boto3 clients
2. **Add WARP VPN sidecar** to face-ocr-api pod for network access to 172.31.30.225
3. **Create `profile_matches` table** in PostgreSQL for persisting match results
4. **Integrate into v3 pipeline** — after ChromaDB step, before strip_base64
5. **Add `/v3/identity/{job_id}/{cluster_id}/profiles` endpoint** for frontend
6. **Test with real video** — match detected faces against 1M+ profiles

---

*End of Report*
